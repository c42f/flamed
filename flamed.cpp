// Copyright (C) 2011, Chris Foster and the other authors and contributors.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the software's owners nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// (This is the New BSD license)

#include <GL/glew.h>
#include <GL/gl.h>

#include "flamed.h"
#include "compute_flames.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <math.h>
#include <float.h>

#include <QtGui/QApplication>
#include <QtGui/QKeyEvent>
#include <QtOpenGL/QGLFramebufferObject>
#include <QtOpenGL/QGLShader>
#include <QtOpenGL/QGLShaderProgram>
#include <QtCore/QTimer>
#include <QtCore/QFileInfo>


#if 0
// Deduced from matplotlib's gist_heat colormap
static void heatmap(double x, float& r, float& g, float& b)
{
    x = std::min(std::max(0.0, x), 1.0); // clamp
    r = std::min(1.0, x/0.7);
    g = std::max(0.0, (x - 0.477)/(1 - 0.477));
    b = std::max(0.0, (x - 0.75)/(1 - 0.75));
}

static void printGlError(const char* tag)
{
    GLenum err = glGetError();
#define ERR_PRN(code)               \
    if(err == code)                 \
        std::cout << tag << " " << #code "\n"
    ERR_PRN(GL_INVALID_ENUM);
    ERR_PRN(GL_INVALID_VALUE);
    ERR_PRN(GL_INVALID_OPERATION);
#undef ERR_PRN
}
#endif


QString uniqueIndexedFileName(const QString& nameTemplate)
{
    int idx = 0;
    QString fName;
    while(true)
    {
        fName = nameTemplate.arg(idx, 3, 10, QChar('0'));
        if(!QFileInfo(fName).exists())
            return fName;
        ++idx;
    }
}


shared_ptr<FlameMaps> FlameViewWidget::initMaps()
{
    shared_ptr<FlameMaps> maps(new FlameMaps());

    maps->finalMap.preMap.m *= 0.5;

    FlameMapping defaultMap;
    defaultMap.preMap.m = M22f(1);
    defaultMap.postMap.m = M22f(1);
    defaultMap.variation = 0;
    maps->maps.resize(2, defaultMap);

    maps->maps[0].preMap.m = M22f(1);
    maps->maps[0].postMap.m = M22f(1);
    maps->maps[0].col = C3f(0,0,1);
    maps->maps[0].variation = 0;

    return maps;
}


//------------------------------------------------------------------------------
FlameViewWidget::FlameViewWidget()
    : m_flameMaps(initMaps()),
    m_undoList(),
    m_redoList(),
    m_useGpu(true),
    m_editMaps(false),
    m_editMode(Mode_Translate),
    m_mapToEdit(0),
    m_editPreTransform(true),
    m_lastPos(),
    m_invPick(1),
    m_frameTimer(),
    m_nPasses(0)
{
    m_undoList.push_back(shared_ptr<FlameMaps>(new FlameMaps(*m_flameMaps)));

    // Install timer with zero timeout to continuously insert extra points into
    // the FBO.
    m_frameTimer = new QTimer(this);
    m_frameTimer->setSingleShot(false);
    m_frameTimer->start(0);
    connect(m_frameTimer, SIGNAL(timeout()), this, SLOT(updateGL()));
}


void FlameViewWidget::load(const char* fileName)
{
    std::ifstream inFile(fileName);
    if(!m_flameMaps->load(inFile))
        std::cerr << "failed to load test.flamed\n";
}


void FlameViewWidget::save(const char* fileName)
{
    std::ofstream outFile(fileName);
    m_flameMaps->save(outFile);
}


void FlameViewWidget::initializeGL()
{
    if(glewInit() != GLEW_OK)
        QApplication::quit();

    m_hdriProgram.reset(new QGLShaderProgram);
    if(!m_hdriProgram->addShaderFromSourceFile(QGLShader::Fragment, "../hdri.glsl"))
        std::cout << "Shader compilation failed:\n"
                  << m_hdriProgram->log().toStdString() << "\n";
    if(!m_hdriProgram->link())
        std::cout << "Shader linking failed:\n"
                  << m_hdriProgram->log().toStdString() << "\n";

    m_pointRenderProgram.reset(new QGLShaderProgram);
    if(!m_pointRenderProgram->addShaderFromSourceFile(QGLShader::Fragment,
                                                     "../point_accum.glsl"))
        std::cout << "Shader compilation failed:\n"
                  << m_pointRenderProgram->log().toStdString() << "\n";
    if(!m_pointRenderProgram->link())
        std::cout << "Shader linking failed:\n"
                  << m_pointRenderProgram->log().toStdString() << "\n";

    m_pointAccumFBO.reset(new QGLFramebufferObject(size(), QGLFramebufferObject::NoAttachment,
                                                   GL_TEXTURE_2D, GL_RGBA32F));

    initCuda();
    if(m_useGpu)
        m_flameEngine.reset(new GPUFlameEngine());
    else
        m_flameEngine.reset(new CPUFlameEngine());

    const int batchSize = 2000000;
    m_ifsPoints.reset(new PointVBO(batchSize));
}


void FlameViewWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    m_pointAccumFBO.reset(
        new QGLFramebufferObject(size(), QGLFramebufferObject::NoAttachment,
                                 GL_TEXTURE_2D, GL_RGBA32F)
    );
    clearAccumulator();
}


void FlameViewWidget::paintGL()
{
    if(m_flameMaps->maps.empty())
        return;

    // Draw fractal
    m_flameEngine->generate(m_ifsPoints.get(), *m_flameMaps);
    ++m_nPasses;

    // Render all points into framebuffer object
    m_pointAccumFBO->bind();
    loadScreenCoords();
    // Additive blending for points
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
//    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_POINT_SPRITE);
    glColor3f(0.5,0.5,0.5);
    glPointSize(1);
    m_pointRenderProgram->bind();
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    m_ifsPoints->bind();
    glVertexPointer(2, GL_FLOAT, sizeof(IFSPoint), 0);
    glColorPointer (3, GL_FLOAT, sizeof(IFSPoint), ((char*)0) + sizeof(V2f));
    glDrawArrays(GL_POINTS, 0, m_ifsPoints->size());
    m_ifsPoints->release();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    m_pointRenderProgram->release();
    m_pointAccumFBO->release();

    glDisable(GL_BLEND);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    m_hdriProgram->bind();
    // Bind FBO to texture unit & set shader params.
    GLint texUnit = 0;
    glActiveTexture(GL_TEXTURE0 + texUnit);
    m_hdriProgram->setUniformValue("tex", texUnit);
    float pointDens = float(width()*height()) / (m_nPasses*m_ifsPoints->size());
    m_hdriProgram->setUniformValue("hdriExposure", m_flameMaps->hdrExposure*pointDens);
    m_hdriProgram->setUniformValue("hdriPow", m_flameMaps->hdrPow);
    glBindTexture(GL_TEXTURE_2D, m_pointAccumFBO->texture());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(-1, -1);
        glTexCoord2f(1, 0);
        glVertex2f(1, -1);
        glTexCoord2f(1, 1);
        glVertex2f(1, 1);
        glTexCoord2f(0, 1);
        glVertex2f(-1, 1);
    glEnd();
    m_hdriProgram->release();

    if(m_editMaps)
        drawMaps(m_flameMaps.get());
    // Debug: for timing.
//    if(m_nPasses*m_ifsPoints->size() > 400000000)
//        close();
}


void FlameViewWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->key() == Qt::Key_Plus)
        m_flameMaps->hdrExposure *= 1.5;
    else if(event->key() == Qt::Key_Minus)
        m_flameMaps->hdrExposure /= 1.5;
    else if(event->key() == Qt::Key_BracketRight)
        m_flameMaps->hdrPow *= 1.5;
    else if(event->key() == Qt::Key_BracketLeft)
        m_flameMaps->hdrPow /= 1.5;
    else if(event->key() == Qt::Key_E)
        m_editMaps = !m_editMaps;
    else if(event->key() == Qt::Key_A)
        m_editPreTransform = !m_editPreTransform;
    else if(event->key() >= Qt::Key_1 && event->key() <= Qt::Key_9)
        m_mapToEdit = std::min((int)m_flameMaps->maps.size()-1,
                               event->key() - Qt::Key_1);
    else if(event->key() == Qt::Key_T)
        m_editMode = Mode_Translate;
    else if(event->key() == Qt::Key_S)
        m_editMode = Mode_Scale;
    else if(event->key() == Qt::Key_R)
        m_editMode = Mode_Rotate;
    else if(event->key() == Qt::Key_Tab)
    {
        if(event->modifiers() | Qt::ControlModifier)
            ++m_mapToEdit;
        else
            --m_mapToEdit;
        m_mapToEdit = (m_mapToEdit + m_flameMaps->maps.size()) %
                      m_flameMaps->maps.size();
    }
    else if(event->key() == Qt::Key_P)
    {
        // get file name
        QString fName = uniqueIndexedFileName("../output/output%1.png");
        grabFrameBuffer().save(fName);
    }
    else if(event->key() == Qt::Key_L)
    {
        // Test - save or load fractal flame
        if(event->modifiers() == Qt::ControlModifier)
        {
            std::ofstream outFile("test.flamed");
            m_flameMaps->save(outFile);
        }
        else
        {
            std::ifstream inFile("test.flamed");
            if(!m_flameMaps->load(inFile))
                std::cout << "failed to load test.flamed\n";
            clearAccumulator();
            m_frameTimer->start(0);
        }
    }
    else if(event->key() == Qt::Key_Escape)
        close();
    else if(event->key() == Qt::Key_Z &&
            event->modifiers() == Qt::ControlModifier)
    {
        if(m_undoList.size() > 1)
        {
            m_redoList.push_back(m_undoList.back());
            m_undoList.pop_back();
            *m_flameMaps = *m_undoList.back();
            clearAccumulator();
        }
    }
    else if(event->key() == Qt::Key_Z && event->modifiers() ==
            (Qt::ControlModifier | Qt::ShiftModifier))
    {
        if(!m_redoList.empty())
        {
            m_undoList.push_back(m_redoList.back());
            m_redoList.pop_back();
            *m_flameMaps = *m_undoList.back();
            clearAccumulator();
        }
    }
    else if(event->key() == Qt::Key_G &&
            event->modifiers() == Qt::ControlModifier)
    {
        // Switch between GPU & CPU renderers
        if(!m_useGpu)
        {
            std::cout << "Using GPU\n";
            makeCurrent();
            m_flameEngine.reset(new GPUFlameEngine());
            m_useGpu = true;
        }
        else
        {
            std::cout << "Using CPU\n";
            m_flameEngine.reset(new CPUFlameEngine());
            m_useGpu = false;
        }
    }
    else if(event->key() == Qt::Key_F11)
    {
        // toggle fullscreen
        setWindowState(windowState() ^ Qt::WindowFullScreen);
    }
    else
        event->ignore();
}


void FlameViewWidget::mousePressEvent(QMouseEvent* event)
{
    m_lastPos = event->pos();
    float aspect = float(width())/height();
    V2f mousePos = V2f( 2*float(m_lastPos.x())/height() - aspect,
                       -2*float(m_lastPos.y())/height() + 1);
    if(event->button() == Qt::RightButton)
    {
        float closestDist = FLT_MAX;
        // Find centre of nearest transform & use that one.
        for(int i = 0; i < (int)m_flameMaps->maps.size(); ++i)
        {
            V2f c = m_flameMaps->fullMap(V2f(0), i);
            float d2 = (c - mousePos).length2();
            if(d2 < closestDist)
            {
                closestDist = d2;
                m_mapToEdit = i;
            }
        }
    }
    // Map a grid from screen coordinates back into screen coordinates &
    // choose the closest resulting point to the mouse position.
    float closestDist = FLT_MAX;
    const int N = 100;
    float scale = 1;
//    aspect = 1;
//    for(int j = 0; j <= N; ++j)
//    for(int i = 0; i <= N; ++i)
//    {
//        V2f p = scale*V2f(aspect*(2.0*i/N - 1), 2.0*j/N - 1);
//        V2f pMap = m_flameMaps->fullMap(p, m_mapToEdit);
//        float d2 = (pMap - mousePos).length2();
//        if(d2 < closestDist)
//        {
//            closestDist = d2;
//            m_invPick = p;
//        }
//    }
    for(int i = 0; i < N; ++i)
    {
        float theta = 2*M_PI*i/N;
        V2f p = scale*V2f(cos(theta),  sin(theta));
        V2f pMap = m_flameMaps->fullMap(p, m_mapToEdit);
        float d2 = (pMap - mousePos).length2();
        if(d2 < closestDist)
        {
            closestDist = d2;
            m_invPick = p;
        }
    }
    updateGL();
}


void FlameViewWidget::mouseMoveEvent(QMouseEvent* event)
{
    QPointF delta = event->pos() - m_lastPos;
    FlameMapping& map = m_flameMaps->maps[m_mapToEdit];
    float aspect = float(width())/height();
    float dx = aspect*delta.x()/width();
    float dy = -float(delta.y())/height();
    V2f df = 4*V2f(dx, dy);
    switch(m_editMode)
    {
        case Mode_Translate: map.translate(m_invPick, df, m_editPreTransform); break;
        case Mode_Scale:     map.scale    (m_invPick, df, m_editPreTransform); break;
        case Mode_Rotate:    map.rotate   (m_invPick, df, m_editPreTransform); break;
    }
    m_lastPos = event->pos();
    clearAccumulator();
}


void FlameViewWidget::mouseReleaseEvent(QMouseEvent* event)
{
    m_redoList.clear();
    m_undoList.push_back(m_flameMaps);
    m_flameMaps.reset(new FlameMaps(*m_flameMaps));
}


QSize FlameViewWidget::sizeHint() const
{
    return QSize(1000, 750);
}


void FlameViewWidget::loadScreenCoords() const
{
    glLoadIdentity();
    float aspect = float(width())/height();
    glScalef(1/aspect, 1, 1);
}


void FlameViewWidget::drawMaps(const FlameMaps* flameMaps)
{
    loadScreenCoords();

    const std::vector<FlameMapping>& maps = flameMaps->maps;

    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float scale = 1;
    for(int k = 0; k < (int)maps.size(); ++k)
    {
        C3f c = maps[k].col;
        if(k != m_mapToEdit)
            c = 0.5*c;
        glColor(c);
        glBegin(GL_LINES);
        const int N = 100;
        for(int i = 0; i < N; ++i)
        {
            float theta = 2*M_PI*i/N;
            float theta1 = 2*M_PI*(i+1)/N;
            glVertex(flameMaps->fullMap(scale*V2f(cos(theta),  sin(theta)), k));
            glVertex(flameMaps->fullMap(scale*V2f(cos(theta1), sin(theta1)), k));
        }
        for(int i = 0; i < N; ++i)
        {
            glVertex(flameMaps->fullMap(scale*V2f(2.0*i/N - 1,     0), k));
            glVertex(flameMaps->fullMap(scale*V2f(2.0*(i+1)/N - 1, 0), k));
            glVertex(flameMaps->fullMap(scale*V2f(0, 2.0*i/N - 1    ), k));
            glVertex(flameMaps->fullMap(scale*V2f(0, 2.0*(i+1)/N - 1), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(2.0*i/N - 1),     -1), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(2.0*(i+1)/N - 1), -1), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(1), 2.0*i/N - 1    ), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(1), 2.0*(i+1)/N - 1), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(-1), 2.0*i/N - 1    ), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(-1), 2.0*(i+1)/N - 1), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(2.0*i/N - 1    ), 1), k));
//            glVertex(flameMaps->fullMap(scale*V2f(2*(2.0*(i+1)/N - 1), 1), k));
        }
        glEnd();
    }

    // Plot position if transformation handle
    V2f res = flameMaps->fullMap(m_invPick, m_mapToEdit);
    glPointSize(10);
    glColor(C3f(1));
    glBegin(GL_POINTS);
        glVertex(res);
    glEnd();

    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
}


void FlameViewWidget::clearAccumulator()
{
    m_pointAccumFBO->bind();
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);
    m_pointAccumFBO->release();
    m_nPasses = 0;
}


int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    FlameViewWidget view;
    view.show();

    // Assume any command line arguments are files to load.
    if(argc > 1)
        view.load(argv[1]);

    return app.exec();
}


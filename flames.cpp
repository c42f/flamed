#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "flames.h"
#include "compute_flames.h"

#include <algorithm>
#include <iostream>
#include <math.h>

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


shared_ptr<FlameMaps> FlameViewWidget::initMaps()
{
    shared_ptr<FlameMaps> maps(new FlameMaps());

    maps->maps.resize(3);
    maps->maps[0].col = C3f(1,0,0);
    maps->maps[0].variation = 0;

    maps->maps[1].col = C3f(0,1,0);
    maps->maps[1].variation = 0;

    maps->maps[2].preMap.m = M22f(0.5);
    maps->maps[2].col = C3f(1);
    maps->maps[2].variation = 0;

    return maps;
}


//------------------------------------------------------------------------------
FlameViewWidget::FlameViewWidget()
    : m_flameMaps(initMaps()),
    m_undoList(),
    m_redoList(),
    m_editMaps(true),
    m_editMode(Mode_Translate),
    m_mapToEdit(0),
    m_editPreTransform(true),
    m_lastPos(),
    m_invPick(1),
    m_screenYMax(2),
    m_frameTimer(),
    m_hdriExposure(1),
    m_hdriPow(1),
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


void FlameViewWidget::initializeGL()
{
    initCuda();
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

    m_ifsPoints.reset(new PointVBO(100000));
}


void FlameViewWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    m_pointAccumFBO.reset(
        new QGLFramebufferObject(size(), QGLFramebufferObject::NoAttachment,
                                 GL_TEXTURE_2D, GL_RGBA32F)
    );
    clearAccumulator();
    m_pickerFBO.reset(
        new QGLFramebufferObject(/*1, 1*/size(), QGLFramebufferObject::NoAttachment,
                                 GL_TEXTURE_2D, GL_RG32F)
    );
}


void FlameViewWidget::paintGL()
{
    computeFractalFlameGPU(m_ifsPoints.get(), *m_flameMaps);
    ++m_nPasses;

    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

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
//    glEnable(GL_TEXTURE_2D);
    // Bind FBO to texture unit & set shader params.
    GLint texUnit = 0;
    glActiveTexture(GL_TEXTURE0 + texUnit);
    m_hdriProgram->setUniformValue("tex", texUnit);
    float pointDens = float(width()*height()) / (m_nPasses*m_ifsPoints->size());
    m_hdriProgram->setUniformValue("hdriExposure", m_hdriExposure*pointDens);
    m_hdriProgram->setUniformValue("hdriPow", m_hdriPow);
    glBindTexture(GL_TEXTURE_2D, m_pointAccumFBO->texture());
//    glBindTexture(GL_TEXTURE_2D, m_pickerFBO->texture());
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
//    glDisable(GL_TEXTURE_2D);
    m_hdriProgram->release();

    if(m_editMaps)
        drawMaps(m_flameMaps.get());
    // Debug: for timing.
//    if(m_nPasses*m_ifsPoints->size() > 40000000)
//        close();
}


void FlameViewWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->key() == Qt::Key_Plus)
        m_hdriExposure *= 1.5;
    else if(event->key() == Qt::Key_Minus)
        m_hdriExposure /= 1.5;
    else if(event->key() == Qt::Key_BracketRight)
        m_hdriPow *= 1.5;
    else if(event->key() == Qt::Key_BracketLeft)
        m_hdriPow /= 1.5;
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
        QString nameTemplate = QString("output%1.png");
        int idx = 0;
        QString fName;
        while(true)
        {
            fName = nameTemplate.arg(idx, 3, 10, QChar('0'));
            if(!QFileInfo(fName).exists())
                break;
            ++idx;
        }
        grabFrameBuffer().save(fName);
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
    event->ignore();
//    std::cout << m_hdriExposure << "  " << m_hdriPow << "\n";
}

void FlameViewWidget::mousePressEvent(QMouseEvent* event)
{
    m_lastPos = event->pos();

    // Compute the point which maps into the clicked location when applying the
    // currently selected mapping.
    //
    // TODO: Make this suck less (use a tiny restricted viewport for efficiency)!
    m_pickerFBO->bind();

    glPushAttrib(GL_VIEWPORT_BIT | GL_TRANSFORM_BIT | GL_COLOR_BUFFER_BIT);
//    glViewport(0, 0, 1, 1);
    glClearColor(-1,-1,-1,1);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
//    glLoadIdentity();
//    int x = event->pos().x();
//    int y = event->pos().y();
//    glOrtho(x, x+1, y+1, y, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
//    glLoadIdentity();
//    float aspect = float(width())/height();
//    glScalef(1/(m_screenYMax*aspect), 1/m_screenYMax, 1);
    loadScreenCoords();

//    drawMaps(m_flameMaps.get());
    // Render out the current map
    const FlameMapping& m = m_flameMaps->maps[m_mapToEdit];
    const float maxDrawArea = 2;
    const int N = 30;
    glBegin(GL_QUADS);
    for(int j = 0; j < N; ++j)
    for(int i = 0; i < N; ++i)
    {
        V2f p0 = m.map(V2f(2.0*i/N     - 1, 2.0*j/N     - 1));
        V2f p1 = m.map(V2f(2.0*(i+1)/N - 1, 2.0*j/N     - 1));
        V2f p2 = m.map(V2f(2.0*(i+1)/N - 1, 2.0*(j+1)/N - 1));
        V2f p3 = m.map(V2f(2.0*i/N     - 1, 2.0*(j+1)/N - 1));
        V2f d1 = p1 - p0;
        V2f d2 = p2 - p0;
        V2f d3 = p3 - p0;
        float area = 0.5*fabs(cross(d2, d1) + cross(d3, d2));
        if(area < maxDrawArea)
        {
            glColor3f(float(i)/N, float(j)/N, 0);
            glVertex(p0);
            glColor3f(float(i+1)/N, float(j)/N, 0);
            glVertex(p1);
            glColor3f(float(i+1)/N, float(j+1)/N, 0);
            glVertex(p2);
            glColor3f(float(i)/N, float(j+1)/N, 0);
            glVertex(p3);
        }
    }
    glEnd();

    glPopAttrib();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    float* fInv = new float[3*width()*height()];
//    float fInv[3] = {0};
    glBindTexture(GL_TEXTURE_2D, m_pickerFBO->texture());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, fInv);
//    m_invPick.x = fInv[0];
//    m_invPick.y = fInv[1];
    float r = fInv[3*(width()*(height() - event->y()) + event->x())];
    float g = fInv[3*(width()*(height() - event->y()) + event->x()) + 1];
    if(r >= 0 && g >= 0)
    {
        m_invPick.x = 2*r - 1;
        m_invPick.y = 2*g - 1;
    }
    delete[] fInv;

//    if(m_editMode == Mode_Scale)
//    {
//        m_invPick.x = m_invPick.x > 0 ? 1 : -1;
//        m_invPick.y = m_invPick.y > 0 ? 1 : -1;
//    }

    m_pickerFBO->release();
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
    return QSize(640, 480);
}


void FlameViewWidget::loadScreenCoords() const
{
    glLoadIdentity();
    float aspect = float(width())/height();
    glScalef(1/(m_screenYMax*aspect), 1/m_screenYMax, 1);
}


void FlameViewWidget::drawMaps(const FlameMaps* flameMaps)
{
    loadScreenCoords();

    const std::vector<FlameMapping>& maps = flameMaps->maps;

    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    for(int k = 0; k < (int)maps.size(); ++k)
    {
        const FlameMapping& m = maps[k];
        C3f c = m.col;
        if(k != m_mapToEdit)
            c = 0.5*c;
        glColor(c);
        glBegin(GL_LINES);
        const int N = 30;
        for(int j = 0; j <= N; j+=N)
        for(int i = 1; i <= N; ++i)
        {
            V2f p0(2.0*(i-1)/N - 1, 2.0*j/N - 1);
            V2f p1(2.0*i    /N - 1, 2.0*j/N - 1);
            p0 = m.map(p0);
            p1 = m.map(p1);
            glVertex2f(p0.x, p0.y);
            glVertex2f(p1.x, p1.y);
        }
        for(int j = 1; j <= N; ++j)
        for(int i = 0; i <= N; i+=N)
        {
            V2f p0(2.0*i/N - 1, 2.0*(j-1)/N - 1);
            V2f p1(2.0*i/N - 1, 2.0*j    /N - 1);
            p0 = m.map(p0);
            p1 = m.map(p1);
            glVertex2f(p0.x, p0.y);
            glVertex2f(p1.x, p1.y);
        }
        glEnd();
    }

    // Plot position if transformation handle
    const FlameMapping& m = maps[m_mapToEdit];
    V2f res = m.map(m_invPick);
    glPointSize(10);
    glColor(C3f(1));
    glBegin(GL_POINTS);
        glVertex2f(res.x, res.y);
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

    return app.exec();
}


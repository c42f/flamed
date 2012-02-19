#include "flames.h"

#include <algorithm>
#include <iostream>
#include <complex>
#include <vector>
#include <float.h>
#include <math.h>

#include <OpenEXR/ImathColor.h>

#include <QtGui/QApplication>
#include <QtOpenGL/QGLFramebufferObject>
#include <QtOpenGL/QGLShader>
#include <QtOpenGL/QGLShaderProgram>
#include <QtCore/QTimer>

#if 0
// Deduced from matplotlib's gist_heat colormap
static void heatmap(double x, float& r, float& g, float& b)
{
    x = std::min(std::max(0.0, x), 1.0); // clamp
    r = std::min(1.0, x/0.7);
    g = std::max(0.0, (x - 0.477)/(1 - 0.477));
    b = std::max(0.0, (x - 0.75)/(1 - 0.75));
}
#endif


template<int base>
inline float radicalInverse(int n)
{
    float invbase = 1.0 / base;
    float scale = invbase;
    float value = 0;
    for (; n != 0; n /= base)
    {
        value += (n % base) * scale;
        scale *= invbase;
    }
    return value;
}


//------------------------------------------------------------------------------
FlameViewWidget::FlameViewWidget()
    : m_bbox(-2,-2,4,4),
    m_frameTimer()
{
    // Install timer with zero timeout to continuously insert extra points into
    // the FBO.
    m_frameTimer = new QTimer(this);
    m_frameTimer->setSingleShot(false);
    m_frameTimer->start(0);
    connect(m_frameTimer, SIGNAL(timeout()), this, SLOT(updateGL()));
}


void FlameViewWidget::initializeGL()
{
//    GLenum glewErr = glewInit();
//    if(glewErr != GLEW_OK)
//        std::cerr << "GLEW init error " << glewGetErrorString(glewErr) << "\n";

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
}

void FlameViewWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    m_pointAccumFBO.reset(
        new QGLFramebufferObject(size(), QGLFramebufferObject::NoAttachment,
                                 GL_TEXTURE_2D, GL_RGBA32F)
    );
    m_pointAccumFBO->bind();
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);
    m_pointAccumFBO->release();
}

#define SHADER_SOURCE(src) #src

void FlameViewWidget::paintGL()
{
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Render all points into framebuffer object
    m_pointAccumFBO->bind();

    glLoadIdentity();
    float aspect = float(width())/height();
    glScalef(0.5/aspect, 0.5, 1);

    // Additive blending for points
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_POINT_SPRITE);
//    glEnable(GL_POINT_SMOOTH);
    glColor3f(0.5,0.5,0.5);
    glPointSize(1);
    m_pointRenderProgram->bind();
    // Julia set example.
    std::complex<double> z = 0;
    std::complex<double> c(0.4,0.3);
    Imath::C3f c1(1,0,0);
    Imath::C3f c2(0,1,0);
    Imath::C3f col(0);
    int batchSize = 10000;
    std::vector<float> pointPos(3*batchSize);
    std::vector<float> pointCol(3*batchSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    for(int i = 0; i < batchSize; ++i)
    {
        if(float(rand())/RAND_MAX > 0.5)
        {
            z = sqrt(z - c);
            col = 0.5f*(col + c1);
        }
        else
        {
            z = -sqrt(z - c);
            col = 0.5f*(col + c2);
        }
        pointPos[3*i] = real(z); pointPos[3*i+1] = imag(z); pointPos[3*i+2] = 0;
        pointCol[3*i] = col.x;   pointCol[3*i+1] = col.y;   pointCol[3*i+2] = col.z;
    }
    glVertexPointer(3, GL_FLOAT, 3*sizeof(float), &pointPos[0]);
    glColorPointer(3, GL_FLOAT, 3*sizeof(float), &pointCol[0]);
    glDrawArrays(GL_POINTS, 0, batchSize);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    // Set of Halton points
//    glBegin(GL_POINTS);
//    for(int i = 0; i < 10000; ++i)
//    {
//        float x = 2*radicalInverse<2>(i) - 1;
//        float y = 2*radicalInverse<3>(i) - 1;
//        float r = 0, g = 0, b = 0;
//        heatmap(i/10000.0, r, g, b);
//        glColor3d(r, g, b);
//        glVertex2f(x, y);
//    }
//    glEnd();
    m_pointRenderProgram->release();

    m_pointAccumFBO->release();

//    std::cout << (m_pointAccumFBO->format().internalTextureFormat() == GL_RGBA32F) << "\n";

    glDisable(GL_BLEND);
    glLoadIdentity();
    m_hdriProgram->bind();
    // Bind FBO to texture unit & set shader params.
    GLint texUnit = 0;
    glActiveTexture(GL_TEXTURE0 + texUnit);
    m_hdriProgram->setUniformValue("tex", texUnit);
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
}



QSize FlameViewWidget::sizeHint() const
{
    return QSize(640, 480);
}


int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    FlameViewWidget view;
    view.show();

    return app.exec();
}


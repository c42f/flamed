#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "flames.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <float.h>
#include <math.h>

#include <OpenEXR/ImathColor.h>

#include <QtGui/QApplication>
#include <QtGui/QKeyEvent>
#include <QtOpenGL/QGLFramebufferObject>
#include <QtOpenGL/QGLShader>
#include <QtOpenGL/QGLShaderProgram>
#include <QtCore/QTimer>

template<typename VertexT>
class VertexBufferObject
{
    public:
        VertexBufferObject(size_t size)
            : m_id(0),
            m_size(size)
        {
            glGenBuffers(1, &m_id);
            glBindBuffer(GL_ARRAY_BUFFER, m_id);
            glBufferData(GL_ARRAY_BUFFER, m_size*sizeof(VertexT), NULL, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        ~VertexBufferObject()
        {
            glDeleteBuffers(1, &m_id);
        }

        void bind() const
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_id);
        }

        void release() const
        {
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        VertexT* mapBuffer(GLenum access = GL_READ_WRITE)
        {
            bind();
            return (VertexT*) glMapBuffer(GL_ARRAY_BUFFER, access);
        }
        const VertexT* mapBuffer() const
        {
            bind();
            return (VertexT*) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        }

        void unmapBuffer() const
        {
            glUnmapBuffer(GL_ARRAY_BUFFER);
            release();
        }

        GLuint id() { return m_id; }
        size_t size() { return m_size; }

    private:
        GLuint m_id;
        size_t m_size;
};


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


struct IFSPoint
{
    float x, y, z;
    float r, g, b;
};


//------------------------------------------------------------------------------
FlameViewWidget::FlameViewWidget()
    : m_bbox(-2,-2,4,4),
    m_frameTimer(),
    m_hdriExposure(1),
    m_hdriPow(1)
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

    m_ifsPoints.reset(new PointVBO(10000));
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
//    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_POINT_SPRITE);
    glColor3f(0.5,0.5,0.5);
    glPointSize(1);
    m_pointRenderProgram->bind();
    genPoints(m_ifsPoints.get());
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    m_ifsPoints->bind();
    glVertexPointer(3, GL_FLOAT, sizeof(IFSPoint), 0);
    glColorPointer (3, GL_FLOAT, sizeof(IFSPoint), ((char*)0) + 3*sizeof(float));
    glDrawArrays(GL_POINTS, 0, m_ifsPoints->size());
    m_ifsPoints->release();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
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
    m_hdriProgram->setUniformValue("hdriExposure", m_hdriExposure);
    m_hdriProgram->setUniformValue("hdriPow", m_hdriPow);
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


void FlameViewWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->key() == Qt::Key_Plus)
        m_hdriExposure /= 1.5;
    else if(event->key() == Qt::Key_Minus)
        m_hdriExposure *= 1.5;
    else if(event->key() == Qt::Key_BracketRight)
        m_hdriPow *= 1.5;
    else if(event->key() == Qt::Key_BracketLeft)
        m_hdriPow /= 1.5;
    else if(event->key() == Qt::Key_Escape)
        close();
//    std::cout << m_hdriExposure << "  " << m_hdriPow << "\n";
}


QSize FlameViewWidget::sizeHint() const
{
    return QSize(640, 480);
}


void FlameViewWidget::genPoints(PointVBO* points)
{
    IFSPoint* ptData = points->mapBuffer(GL_WRITE_ONLY);
    // Julia set example.
    complex z = 0;
    complex c(0.4,0.3);
    Imath::C3f c1(1,0,0);
    Imath::C3f c2(0,1,0);
    Imath::C3f col(0);
    int batchSize = points->size();
    for(int i = 0; i < batchSize; ++i, ++ptData)
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
        ptData->x = real(z); ptData->y = imag(z); ptData->z = 0;
        ptData->r = col.x;   ptData->g = col.y;   ptData->b = col.z;
    }

    // Set of Halton points
//    for(int i = 0; i < (int)points->size(); ++i, ++ptData)
//    {
//        ptData->x = 2*radicalInverse<2>(i) - 1;
//        ptData->y = 2*radicalInverse<3>(i) - 1;
//        ptData->z = 0;
//        heatmap(double(i)/points->size(), ptData->r, ptData->g, ptData->b);
//    }
    points->unmapBuffer();
}

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    FlameViewWidget view;
    view.show();

    return app.exec();
}


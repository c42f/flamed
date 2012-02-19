#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "flames.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <float.h>
#include <math.h>

#include <QtGui/QApplication>
#include <QtGui/QKeyEvent>
#include <QtOpenGL/QGLFramebufferObject>
#include <QtOpenGL/QGLShader>
#include <QtOpenGL/QGLShaderProgram>
#include <QtCore/QTimer>


// Thin wrapper around OpenGL vertex buffer object
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


// Simplistic 3 component color
struct C3f
{
    float x,y,z;
    C3f() {}
    C3f(float x, float y, float z) : x(x), y(y), z(z) {}
    C3f(float v) : x(v), y(v), z(v) {}
};
inline C3f operator+(const C3f& c1, const C3f& c2)
{
    return C3f(c1.x + c2.x, c1.y + c2.y, c1.z + c2.z);
}
inline C3f operator*(float a, const C3f& c)
{
    return C3f(c.x*a, c.y*a, c.z*a);
}
inline C3f operator*(const C3f& c, float a)
{
    return a*c;
}


// Point data for IFS mapping
struct IFSPoint
{
    float x, y;
    C3f col;
};


// Mapping
struct FlameMapping
{
    float a,b,c,d,e,f;
    C3f col;

    FlameMapping() : a(1), b(0), c(0), d(0), e(1), f(0), col(1) {}

    void map(float& x, float& y) const
    {
        float x0 = x, y0 = y;
        x = a*x0 + b*y0 + c;
        y = d*x0 + e*y0 + f;
        x0 = x; y0 = y;
//        float s = 1/(x0*x0 + y0*y0);
//        x = s*x0;
//        y = s*y0;
//        float s = exp(x0 - 1);
//        x = s*cos(M_PI*y0);
//        y = s*sin(M_PI*y0);
//        x = sin(x0)/cos(y0);
//        y = tan(y0);
        float s = 2/(sqrt(x0*x0 + y0*y0) + 1);
        x = y0*s;
        y = x0*s;
    }
};


struct FlameMaps
{
    std::vector<FlameMapping> maps;
};

//------------------------------------------------------------------------------
FlameViewWidget::FlameViewWidget()
    : m_flameMaps(new FlameMaps()),
    m_editMaps(true),
    m_editMode(Mode_Translate),
    m_mapToEdit(0),
    m_lastPos(),
    m_bbox(-2,-2,4,4),
    m_frameTimer(),
    m_hdriExposure(1),
    m_hdriPow(1),
    m_nPasses(0)
{
    // Install timer with zero timeout to continuously insert extra points into
    // the FBO.
    m_frameTimer = new QTimer(this);
    m_frameTimer->setSingleShot(false);
    m_frameTimer->start(0);
    connect(m_frameTimer, SIGNAL(timeout()), this, SLOT(updateGL()));
    m_flameMaps->maps.resize(2);
    m_flameMaps->maps[0].a = 0.6;
    m_flameMaps->maps[0].e = 0.6;
    m_flameMaps->maps[0].c = 0.5;
    m_flameMaps->maps[0].f = 0.5;
    m_flameMaps->maps[0].col = C3f(1,0,0);

    m_flameMaps->maps[1].a = 0.4;
    m_flameMaps->maps[1].b = 0.4;
    m_flameMaps->maps[1].d = -0.5;
    m_flameMaps->maps[1].e = 0.5;
    m_flameMaps->maps[1].c = -0.5;
    m_flameMaps->maps[1].col = C3f(0,0,1);
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
    clearAccumulator();
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
    ++m_nPasses;
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    m_ifsPoints->bind();
    glVertexPointer(2, GL_FLOAT, sizeof(IFSPoint), 0);
    glColorPointer (3, GL_FLOAT, sizeof(IFSPoint), ((char*)0) + 2*sizeof(float));
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
    m_hdriProgram->setUniformValue("hdriExposure", m_hdriExposure*m_nPasses);
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

    if(m_editMaps)
        drawMaps(m_flameMaps.get());
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
    else if(event->key() == Qt::Key_E)
        m_editMaps = !m_editMaps;
    else if(event->key() >= Qt::Key_1 && event->key() <= Qt::Key_9)
        m_mapToEdit = std::min((int)m_flameMaps->maps.size()-1, event->key() - Qt::Key_1);
    else if(event->key() == Qt::Key_T)
        m_editMode = Mode_Translate;
    else if(event->key() == Qt::Key_S)
        m_editMode = Mode_Scale;
    else if(event->key() == Qt::Key_R)
        m_editMode = Mode_Rotate;
    else if(event->key() == Qt::Key_Escape)
        close();
    event->ignore();
//    std::cout << m_hdriExposure << "  " << m_hdriPow << "\n";
}


void FlameViewWidget::mousePressEvent(QMouseEvent* event)
{
    m_lastPos = event->pos();
}


void FlameViewWidget::wheelEvent(QWheelEvent* event)
{
    m_mapToEdit += event->delta() > 0 ? 1 : -1;
    m_mapToEdit = (m_mapToEdit + m_flameMaps->maps.size()) % m_flameMaps->maps.size();
}


void FlameViewWidget::mouseMoveEvent(QMouseEvent* event)
{
    QPointF delta = event->pos() - m_lastPos;
    FlameMapping& map = m_flameMaps->maps[m_mapToEdit];
    float aspect = float(width())/height();
    float dx = aspect*delta.x()/width();
    float dy = -float(delta.y())/height();
    switch(m_editMode)
    {
        case Mode_Translate:
            map.c += 4.0*dx;
            map.f += 4.0*dy;
            break;
        case Mode_Scale:
//            map.a += 4*dx;
//            map.b += 4*dy;
//            map.d += 4*dx;
//            map.e += 4*dy;
            map.a *= (1 + 2*dx);
            map.b *= (1 + 2*dx);
            map.d *= (1 + 2*dy);
            map.e *= (1 + 2*dy);
            break;
        case Mode_Rotate:
            break;
    }
    m_lastPos = event->pos();
    clearAccumulator();
}


QSize FlameViewWidget::sizeHint() const
{
    return QSize(640, 480);
}


void FlameViewWidget::genPoints(PointVBO* points)
{
    IFSPoint* ptData = points->mapBuffer(GL_WRITE_ONLY);

#if 1
    int nMaps = m_flameMaps->maps.size();
    // Flame fractals!
    float x = 0;
    float y = 0;
    int discard = 20;
    C3f col(1);
    int nPoints = points->size();
    for(int i = -discard; i < nPoints; ++i)
    {
        int mapIdx = rand() % nMaps;
        const FlameMapping& m = m_flameMaps->maps[mapIdx];
        m.map(x, y);
        col = 0.5*(col + m.col);
        if(i >= 0)
        {
            ptData->x = x;
            ptData->y = y;
            ptData->col = col;
            ++ptData;
        }
    }
#endif

#if 0
    // Julia set example.
    complex z = 0;
    complex c(0.4,0.3);
    C3f c1(1,0,0);
    C3f c2(0,1,0);
    C3f col(0);
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
        ptData->x = real(z); ptData->y = imag(z);
        ptData->col = col;
    }
#endif

    // Set of Halton points
#if 0
    for(int i = 0; i < (int)points->size(); ++i, ++ptData)
    {
        ptData->x = 2*radicalInverse<2>(i) - 1;
        ptData->y = 2*radicalInverse<3>(i) - 1;
        heatmap(double(i)/points->size(), ptData->col.x, ptData->col.y, ptData->col.z);
    }
#endif
    points->unmapBuffer();
}


void FlameViewWidget::drawMaps(const FlameMaps* flameMaps)
{
    glLoadIdentity();
    float aspect = float(width())/height();
    glScalef(0.5/aspect, 0.5, 1);

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
            c = 0.2*c;
        glColor3f(c.x, c.y, c.z);
        glBegin(GL_LINES);
        const int N = 10;
        for(int j = 0; j <= N; ++j)
        for(int i = 1; i <= N; ++i)
        {
            float x0 = 2.0*(i-1)/N - 1;
            float x1 = 2.0*i/N - 1;
            float y0 = 2.0*j/N - 1;
            float y1 = y0;
            m.map(x0, y0);
            m.map(x1, y1);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
        }
        for(int j = 1; j <= N; ++j)
        for(int i = 0; i <= N; ++i)
        {
            float x0 = 2.0*i/N - 1;
            float x1 = x0;
            float y0 = 2.0*j/N - 1;
            float y1 = 2.0*(j-1)/N - 1;
            m.map(x0, y0);
            m.map(x1, y1);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
        }
        glEnd();
    }

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


//#include <glew.h>

#include <QtOpenGL/QGLWidget>
#include <tr1/memory>
#include <complex>

typedef std::complex<double> complex;

class QGLShaderProgram;
class QGLFramebufferObject;
class QTimer;
template<typename> class VertexBufferObject;
struct IFSPoint;
typedef VertexBufferObject<IFSPoint> PointVBO;

class FlameViewWidget : public QGLWidget
{
    Q_OBJECT
    public:
        FlameViewWidget();

    protected:
        // GL stuff
        void initializeGL();
        void resizeGL(int w, int h);
        void paintGL();

        // User events
        void keyPressEvent(QKeyEvent* event);
//        void mousePressEvent(QMouseEvent* event);
//        void mouseMoveEvent(QMouseEvent* event);

        QSize sizeHint() const;

    private:
        void genPoints(PointVBO* points);

        std::tr1::shared_ptr<QGLShaderProgram> m_pointRenderProgram;
        std::tr1::shared_ptr<QGLShaderProgram> m_hdriProgram;
        std::tr1::shared_ptr<QGLFramebufferObject> m_pointAccumFBO;
        std::tr1::shared_ptr<PointVBO> m_ifsPoints;

//        QPoint m_lastPos;
        QRectF m_bbox;
        QTimer* m_frameTimer;
        float m_hdriExposure;
        float m_hdriPow;
};


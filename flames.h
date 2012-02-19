//#include <glew.h>

#include <QtOpenGL/QGLWidget>
#include <QtCore/QTimer>
#include <tr1/memory>

class QGLShaderProgram;
class QGLFramebufferObject;

class FlameViewWidget : public QGLWidget
{
    Q_OBJECT
    public:
        FlameViewWidget();

    protected:
        void initializeGL();
        void resizeGL(int w, int h);
        void paintGL();
//        void keyPressEvent(QKeyEvent* event);
//        void paintEvent(QPaintEvent* event);
//        void mousePressEvent(QMouseEvent* event);
//        void mouseMoveEvent(QMouseEvent* event);

        QSize sizeHint() const;

    private:
        void renderImage();

        std::tr1::shared_ptr<QGLShaderProgram> m_pointRenderProgram;
        std::tr1::shared_ptr<QGLShaderProgram> m_hdriProgram;
        std::tr1::shared_ptr<QGLFramebufferObject> m_pointAccumFBO;

//        QPoint m_lastPos;
        QRectF m_bbox;
        QTimer* m_frameTimer;
};


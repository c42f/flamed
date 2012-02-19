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

struct FlameMaps;

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
        void mousePressEvent(QMouseEvent* event);
        void wheelEvent(QWheelEvent* event);
        void mouseMoveEvent(QMouseEvent* event);

        QSize sizeHint() const;

    private:
        enum EditMode
        {
            Mode_Translate,
            Mode_Rotate,
            Mode_Scale
        };

        void genPoints(PointVBO* points);
        void drawMaps(const FlameMaps* flameMaps);
        void clearAccumulator();

        std::tr1::shared_ptr<QGLShaderProgram> m_pointRenderProgram;
        std::tr1::shared_ptr<QGLShaderProgram> m_hdriProgram;
        std::tr1::shared_ptr<QGLFramebufferObject> m_pointAccumFBO;
        std::tr1::shared_ptr<PointVBO> m_ifsPoints;
        std::tr1::shared_ptr<FlameMaps> m_flameMaps;

        bool m_editMaps;
        EditMode m_editMode;
        int m_mapToEdit;
        QPoint m_lastPos;

        QRectF m_bbox;
        QTimer* m_frameTimer;
        float m_hdriExposure;
        float m_hdriPow;
        int m_nPasses;
};


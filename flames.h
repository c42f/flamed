#include <QtOpenGL/QGLWidget>

#include "util.h"

class QGLShaderProgram;
class QGLFramebufferObject;
class QTimer;
class FlameEngine;
template<typename> class VertexBufferObject;
struct IFSPoint;
typedef VertexBufferObject<IFSPoint> PointVBO;

struct FlameMaps;

// Viewer for flame fractals
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
        void mouseMoveEvent(QMouseEvent* event);
        void mouseReleaseEvent(QMouseEvent* event);

        QSize sizeHint() const;

    private:
        enum EditMode
        {
            Mode_Translate,
            Mode_Rotate,
            Mode_Scale
        };

        static shared_ptr<FlameMaps> initMaps();
        void loadScreenCoords() const;
        void drawMaps(const FlameMaps* flameMaps);
        void clearAccumulator();

        shared_ptr<QGLShaderProgram> m_pointRenderProgram;
        shared_ptr<QGLShaderProgram> m_hdriProgram;
        shared_ptr<QGLFramebufferObject> m_pointAccumFBO;
        shared_ptr<QGLFramebufferObject> m_pickerFBO;
        shared_ptr<PointVBO> m_ifsPoints;
        shared_ptr<FlameMaps> m_flameMaps;
        shared_ptr<FlameEngine> m_flameEngine;
        std::vector<shared_ptr<FlameMaps> > m_undoList;
        std::vector<shared_ptr<FlameMaps> > m_redoList;

        bool m_editMaps;
        EditMode m_editMode;
        int m_mapToEdit;
        bool m_editPreTransform;
        QPoint m_lastPos;
        V2f m_invPick;

        float m_screenYMax;
        QTimer* m_frameTimer;
        float m_hdriExposure;
        float m_hdriPow;
        int m_nPasses;
};


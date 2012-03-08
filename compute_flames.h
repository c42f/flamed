#ifndef COMPUTE_FLAME_H_INCLUDED
#define COMPUTE_FLAME_H_INCLUDED

// ugh...
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <vector>

#include <QtOpenGL/qgl.h>

#include "util.h"

// Thin wrapper around OpenGL vertex buffer object.
//
// VertexT must be a POD type
template<typename VertexT>
class VertexBufferObject
{
    public:
        /// Make a vertex buffer containing size objects of type VertexT.
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

        /// Make the buffer into the current buffer referred to by GL vertex
        /// array function calls (eg, glVertexPointer)
        void bind() const
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_id);
        }

        /// Opposite of bind()
        void release() const
        {
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        /// Map buffer into CPU address space
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

        /// Unmap buffer from CPU address space
        void unmapBuffer() const
        {
            glUnmapBuffer(GL_ARRAY_BUFFER);
            release();
        }

        /// Get underlying OpenGL buffer identifier
        GLuint id() { return m_id; }
        /// Number of vertices in array
        size_t size() { return m_size; }

    private:
        GLuint m_id;
        size_t m_size;
};


// Point data for IFS mapping
struct IFSPoint
{
    V2f pos;
    C3f col;
};

typedef VertexBufferObject<IFSPoint> PointVBO;


// Definition of a fractal flame mapping
struct FlameMapping
{
    AffineMap preMap;
    AffineMap postMap;
    float colorSpeed;
    C3f col;
    int variation;

    FlameMapping()
        : preMap(AffineMap::Init),
        postMap(AffineMap::Init),
        colorSpeed(0.5),
        col(1),
        variation(0)
    {}

    GPU_HOSTDEV V2f nonlinearMap(V2f p) const
    {
        float x = p.x;
        float y = p.y;
        switch(variation)
        {
            case 0:
            default:
                return p;
            case 1: { // Sinusoidal
                return V2f(sin(x), sin(y));
            }
            case 2: { // Spherical
                float s = 1/(x*x + y*y);
                return V2f(s*x, s*y);
            }
            case 3: { // Swirl
                float r2 = x*x + y*y;
                float s = sin(r2), c = cos(r2);
                return V2f(x*s - y*c, x*c + y*s);
            }
            case 4: { // Horseshoe
                float r = sqrt(x*x + y*y);
                return 1/r * V2f((x-y)*(x+y), 2*x*y);
            }
            case 5: { // Polar
                float theta = atan2(x,y);
                float r = sqrt(x*x + y*y);
                return V2f(theta/M_PI, r - 1);
            }
            case 6: { // Handkerchief
                float theta = atan2(x,y);
                float r = sqrt(x*x + y*y);
                return V2f(sin(theta + r), cos(theta - r));
            }
            case 7: { // Heart
                float theta = atan2(x,y);
                float r = sqrt(x*x + y*y);
                return r*V2f(sin(theta*r), -cos(theta*r));
            }
            case 18: { // Exponential
                return exp(x - 1) * V2f(cos(M_PI*y), sin(M_PI*y));
            }
            case 27: { // Eyefish
                return 2/(sqrt(x*x + y*y) + 1) * V2f(x, y);
            }
            case 42: { // Tangent
                return V2f(sin(x)/cos(y), tan(y));
            }
        }
    }

    GPU_HOSTDEV V2f map(V2f p) const
    {
        p = preMap.map(p);
        p = nonlinearMap(p);
        p = postMap.map(p);
        return p;
    }

    // Transform the map such that (map_new(p) - map(p)) ~= df where map_new is
    // map() after the transformation, and df is the amount the mouse moved in
    // screen coordinates.
    void translate(V2f p, V2f df, bool editPreTrans);
    void scale(V2f p, V2f df, bool editPreTrans);
    void rotate(V2f p, V2f df, bool editPreTrans);
};


struct FlameMaps
{
    std::vector<FlameMapping> maps;
//    FlameMapping finalMap;
};


void computeFractalFlame(PointVBO* points, const FlameMaps& flameMaps);


void computeFractalFlameGPU(PointVBO* points, const FlameMaps& flameMaps);

void initCuda();

#endif

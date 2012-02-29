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
    M22f m;
    V2f c;
    C3f col;
    int variation;

    FlameMapping() : m(1), c(0), col(1), variation(0) {}

    V2f mapV1(V2f p) const
    {
        float s = 1/(p.x*p.x + p.y*p.y);
        return V2f(s*p.x, s*p.y);
    }

    V2f mapV2(V2f p) const
    {
        float s = exp(p.x - 1);
        return V2f(s*cos(M_PI*p.y), s*sin(M_PI*p.y));
    }

    V2f mapV3(V2f p) const
    {
        return V2f(sin(p.x)/cos(p.y), tan(p.y));
    }

    V2f mapV4(V2f p) const
    {
        float s = 2/(sqrt(p.x*p.x + p.y*p.y) + 1);
        return V2f(p.y*s, p.x*s);
    }

    V2f mapV5(V2f p) const
    {
        return V2f(sin(p.x), sin(p.y));
    }

    V2f map(V2f p) const
    {
        p = m*p + c;
        switch(variation)
        {
            default: return p;
            case 1: return mapV1(p);
            case 2: return mapV2(p);
            case 3: return mapV3(p);
            case 4: return mapV4(p);
            case 5: return mapV5(p);
        }
    }
};


inline M22f transDeriv(FlameMapping m, V2f p)
{
    const float delta = 0.001;
    V2f r0 = m.map(p);
    m.c.x += delta;
    V2f r1 = m.map(p);
    m.c.x -= delta;
    m.c.y += delta;
    V2f r2 = m.map(p);
    V2f drdx = (r1 - r0)/delta;
    V2f drdy = (r2 - r0)/delta;
    return M22f(drdx.x, drdy.x,
                drdx.y, drdy.y);
}

inline M22f scaleDeriv(FlameMapping m, V2f p)
{
    const float delta = 0.001;
    V2f r0 = m.map(p);
    m.m.a *= (1+delta);
    m.m.c *= (1+delta);
    V2f r1 = m.map(p);
    m.m.a /= (1+delta);
    m.m.c /= (1+delta);
    m.m.b *= (1+delta);
    m.m.d *= (1+delta);
    V2f r2 = m.map(p);
    V2f drdx = (r1 - r0)/delta;
    V2f drdy = (r2 - r0)/delta;
    return M22f(drdx.x, drdy.x,
                drdx.y, drdy.y);
}

inline V2f rotateDeriv(FlameMapping m, V2f p)
{
    const float delta = 0.001;
    V2f r0 = m.map(p);
    m.m = M22f( cos(delta), sin(delta),
               -sin(delta), cos(delta)) * m.m;
    V2f r1 = m.map(p);
    return (r1 - r0)/delta;
}

struct FlameMaps
{
    std::vector<FlameMapping> maps;
};


void computeFractalFlame(PointVBO* points, const FlameMaps& flameMaps);


#endif

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


struct AffineMap
{
    M22f m;
    V2f c;

    enum InitTag { Init };

    AffineMap() {}
    AffineMap(InitTag /*init*/) : m(1), c(0) {}

    V2f map(V2f p) const
    {
        return m*p + c;
    }
};


// Definition of a fractal flame mapping
struct FlameMapping
{
    AffineMap preMap;
    AffineMap postMap;
    C3f col;
    int variation;

    FlameMapping()
        : preMap(AffineMap::Init),
        postMap(AffineMap::Init),
        col(1),
        variation(0)
    {}

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
        p = preMap.map(p);
        switch(variation)
        {
            case 1: p = mapV1(p); break;
            case 2: p = mapV2(p); break;
            case 3: p = mapV3(p); break;
            case 4: p = mapV4(p); break;
            case 5: p = mapV5(p); break;
            default:              break;
        }
        p = postMap.map(p);
        return p;
    }

    // Transformations.
    void translate(V2f p, V2f df, bool editPreTrans);
    void scale(V2f p, V2f df, bool editPreTrans);
    void rotate(V2f p, V2f df, bool editPreTrans);
};


struct FlameMaps
{
    std::vector<FlameMapping> maps;
};


void computeFractalFlame(PointVBO* points, const FlameMaps& flameMaps);


#endif

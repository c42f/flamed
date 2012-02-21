#ifndef COMPUTE_FLAME_H_INCLUDED
#define COMPUTE_FLAME_H_INCLUDED

// ugh...
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <vector>

#include <QtOpenGL/qgl.h>


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


// Simplistic two-component vector class
struct V2f
{
    float x,y;
    V2f() {}
    V2f(float x, float y) : x(x), y(y) {}
    V2f(float v) : x(v), y(v) {}
};
inline V2f operator+(const V2f& c1, const V2f& c2)
{
    return V2f(c1.x + c2.x, c1.y + c2.y);
}
inline V2f operator*(float a, const V2f& c)
{
    return V2f(c.x*a, c.y*a);
}
inline V2f operator*(const V2f& c, float a)
{
    return a*c;
}


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
    float a,b,c,d,e,f;
    C3f col;
    int variation;

    FlameMapping() : a(1), b(0), c(0), d(0), e(1), f(0), col(1), variation(0) {}

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
        p = V2f(a*p.x + b*p.y + c, d*p.x + e*p.y + f);
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


struct FlameMaps
{
    std::vector<FlameMapping> maps;
};


void computeFractalFlame(PointVBO* points, const FlameMaps& flameMaps);


#endif

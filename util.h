// Copyright (C) 2011, Chris Foster and the other authors and contributors.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the software's owners nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// (This is the New BSD license)

#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#ifdef COMPILE_FOR_GPU
#   define GPU_HOSTDEV __host__ __device__
#   define GPU_HOST __host__
#   define GPU_DEV __device__
#else
#   define GPU_HOSTDEV
#endif

#include <math.h>
#include <iostream>

// Compute ceil(real(n)/d) using integers for positive n and d.
template<typename T>
inline T ceildiv(T n, T d)
{
    return (n-T(1))/d + T(1);
}


// Radical inverse function for simple quasi random number generation
template<int base>
GPU_HOSTDEV inline float radicalInverse(int n)
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
    GPU_HOSTDEV C3f() {}
    GPU_HOSTDEV explicit C3f(float v) : x(v), y(v), z(v) {}
    GPU_HOSTDEV C3f(float x, float y, float z) : x(x), y(y), z(z) {}
};
GPU_HOSTDEV inline C3f operator+(const C3f& c1, const C3f& c2)
{
    return C3f(c1.x + c2.x, c1.y + c2.y, c1.z + c2.z);
}
GPU_HOSTDEV inline C3f operator*(float a, const C3f& c)
{
    return C3f(c.x*a, c.y*a, c.z*a);
}
GPU_HOSTDEV inline C3f operator*(const C3f& c, float a)
{
    return a*c;
}
inline void glColor(const C3f& c)
{
    return glColor3f(c.x, c.y, c.z);
}


// Simplistic two-component vector class
struct V2f
{
    float x,y;
    GPU_HOSTDEV V2f() {}
    GPU_HOSTDEV explicit V2f(float v) : x(v), y(v) {}
    GPU_HOSTDEV V2f(float x, float y) : x(x), y(y) {}

    GPU_HOSTDEV float length() const { return sqrt(x*x + y*y); }
    GPU_HOSTDEV float length2() const { return x*x + y*y; }

    GPU_HOSTDEV V2f& operator+=(const V2f& rhs) { x += rhs.x; y += rhs.y; return *this; }
    GPU_HOSTDEV V2f& operator*=(float a) { x *= a; y *= a; return *this; }
};
GPU_HOSTDEV inline V2f operator+(const V2f& c1, const V2f& c2)
{
    return V2f(c1.x + c2.x, c1.y + c2.y);
}
GPU_HOSTDEV inline V2f operator-(const V2f& c1, const V2f& c2)
{
    return V2f(c1.x - c2.x, c1.y - c2.y);
}
GPU_HOSTDEV inline V2f operator*(float a, const V2f& c)
{
    return V2f(c.x*a, c.y*a);
}
GPU_HOSTDEV inline V2f operator*(const V2f& c, float a)
{
    return a*c;
}
GPU_HOSTDEV inline V2f operator/(const V2f& c, float a)
{
    return V2f(c.x/a, c.y/a);
}
GPU_HOSTDEV inline float cross(V2f& a, V2f b)
{
    return a.x*b.y - b.x*a.y;
}
GPU_HOSTDEV inline float dot(const V2f& a, const V2f& b)
{
    return a.x*b.x + a.y*b.y;
}

inline void glVertex(const V2f& v)
{
    return glVertex2f(v.x, v.y);
}


// Simplistic 2x2 matrix
struct M22f
{
    float a,b,
          c,d;
    GPU_HOSTDEV explicit M22f(float x = 1)
        : a(x), b(0), c(0), d(x) {}
    GPU_HOSTDEV explicit M22f(float a, float b, float c, float d)
        : a(a), b(b), c(c), d(d) {}
    GPU_HOSTDEV M22f inv() const;
    GPU_HOSTDEV M22f& operator*=(const M22f m);
    GPU_HOSTDEV M22f& operator*=(float s);
};
GPU_HOSTDEV inline M22f operator*(float s, const M22f& m)
{
    return M22f(s*m.a, s*m.b, s*m.c, s*m.d);
}
GPU_HOSTDEV inline V2f operator*(const M22f& m, const V2f& p)
{
    return V2f(m.a*p.x + m.b*p.y, m.c*p.x + m.d*p.y);
}
GPU_HOSTDEV inline M22f operator*(const M22f& m1, const M22f& m2)
{
    return M22f(m1.a*m2.a + m1.b*m2.c, m1.a*m2.b + m1.b*m2.d,
                m1.c*m2.a + m1.d*m2.c, m1.c*m2.b + m1.d*m2.d);
}
GPU_HOSTDEV inline M22f M22f::inv() const
{
    return 1/(a*d - b*c) * M22f(d, -b, -c, a);
}
GPU_HOSTDEV inline M22f& M22f::operator*=(const M22f m)
{
    *this = (*this)*m;
    return *this;
}
GPU_HOSTDEV inline M22f& M22f::operator*=(float s)
{
    *this = s * (*this);
    return *this;
}


struct AffineMap
{
    M22f m;
    V2f c;

    enum InitTag { Init };

    GPU_HOSTDEV AffineMap() {}
    GPU_HOSTDEV AffineMap(InitTag /*init*/) : m(1), c(0) {}
    GPU_HOSTDEV AffineMap(M22f m, V2f c) : m(m), c(c) {}

    GPU_HOSTDEV V2f map(V2f p) const
    {
        return m*p + c;
    }
};


inline std::ostream& operator<<(std::ostream& out, const AffineMap& map)
{
    out << map.m.a << " " << map.m.b << " "
        << map.m.c << " " << map.m.d << "   "
        << map.c.x << " " << map.c.y;
    return out;
}

inline std::istream& operator>>(std::istream& in, AffineMap& map)
{
    in >> map.m.a >> map.m.b >> map.m.c >> map.m.d
       >> map.c.x >> map.c.y;
    return in;
}

inline std::ostream& operator<<(std::ostream& out, const C3f& col)
{
    out << col.x << " " << col.y << " " << col.z;
    return out;
}

inline std::istream& operator>>(std::istream& in, C3f& col)
{
    in >> col.x >> col.y >> col.z;
    return in;
}

#endif // UTIL_H_INCLUDED

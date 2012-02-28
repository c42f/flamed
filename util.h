#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED


// Simplistic 3 component color
struct C3f
{
    float x,y,z;
    C3f() {}
    explicit C3f(float v) : x(v), y(v), z(v) {}
    C3f(float x, float y, float z) : x(x), y(y), z(z) {}
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
inline void glColor(const C3f& c)
{
    return glColor3f(c.x, c.y, c.z);
}


// Simplistic two-component vector class
struct V2f
{
    float x,y;
    V2f() {}
    explicit V2f(float v) : x(v), y(v) {}
    V2f(float x, float y) : x(x), y(y) {}

    float length() const { return sqrt(x*x + y*y); }

    V2f& operator+=(const V2f& rhs) { x += rhs.x; y += rhs.y; return *this; }
    V2f& operator*=(float a) { x *= a; y *= a; return *this; }
};
inline V2f operator+(const V2f& c1, const V2f& c2)
{
    return V2f(c1.x + c2.x, c1.y + c2.y);
}
inline V2f operator-(const V2f& c1, const V2f& c2)
{
    return V2f(c1.x - c2.x, c1.y - c2.y);
}
inline V2f operator*(float a, const V2f& c)
{
    return V2f(c.x*a, c.y*a);
}
inline V2f operator*(const V2f& c, float a)
{
    return a*c;
}
inline V2f operator/(const V2f& c, float a)
{
    return V2f(c.x/a, c.y/a);
}
inline float cross(V2f& a, V2f b)
{
    return a.x*b.y - b.x*a.y;
}
inline float dot(const V2f& a, const V2f& b)
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
    explicit M22f(float x = 1)
        : a(x), b(0), c(0), d(x) {}
    explicit M22f(float a, float b, float c, float d)
        : a(a), b(b), c(c), d(d) {}
    M22f inv() const;
    M22f& operator*=(const M22f m);
};
inline M22f operator*(float s, const M22f& m)
{
    return M22f(s*m.a, s*m.b, s*m.c, s*m.d);
}
inline V2f operator*(const M22f& m, const V2f& p)
{
    return V2f(m.a*p.x + m.b*p.y, m.c*p.x + m.d*p.y);
}
inline M22f operator*(const M22f& m1, const M22f& m2)
{
    return M22f(m1.a*m2.a + m1.b*m2.c, m1.a*m2.b + m1.b*m2.d,
                m1.c*m2.a + m1.d*m2.c, m1.c*m2.b + m1.d*m2.d);
}
inline M22f M22f::inv() const
{
    return 1/(a*d - b*c) * M22f(d, -b, -c, a);
}
inline M22f& M22f::operator*=(const M22f m)
{
    *this = (*this)*m;
    return *this;
}


#endif // UTIL_H_INCLUDED

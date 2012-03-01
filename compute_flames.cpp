#include "compute_flames.h"


void FlameMapping::translate(V2f p, V2f df, bool editPreTrans)
{
    FlameMapping tmpMap = *this;
    AffineMap& aff = editPreTrans ? tmpMap.preMap : tmpMap.postMap;
    const float delta = 0.001;
    V2f r0 = tmpMap.map(p);
    aff.c.x += delta;
    V2f r1 = tmpMap.map(p);
    aff.c.x -= delta;
    aff.c.y += delta;
    V2f r2 = tmpMap.map(p);
    V2f drdx = (r1 - r0)/delta;
    V2f drdy = (r2 - r0)/delta;
    M22f dfdc(drdx.x, drdy.x,
                drdx.y, drdy.y);
    V2f dc = dfdc.inv()*df;
    const float maxLength = 2;
    if(dc.length() > maxLength)
        dc *= maxLength/dc.length();
    AffineMap& thisAff = editPreTrans ? preMap : postMap;
    thisAff.c += dc;
}

void FlameMapping::scale(V2f p, V2f df, bool editPreTrans)
{
    FlameMapping tmpMap = *this;
    AffineMap& aff = editPreTrans ? tmpMap.preMap : tmpMap.postMap;
    const float delta = 0.001;
    V2f r0 = tmpMap.map(p);
    aff.m.a *= (1+delta);
    aff.m.c *= (1+delta);
    V2f r1 = tmpMap.map(p);
    aff.m.a /= (1+delta);
    aff.m.c /= (1+delta);
    aff.m.b *= (1+delta);
    aff.m.d *= (1+delta);
    V2f r2 = tmpMap.map(p);
    V2f drdx = (r1 - r0)/delta;
    V2f drdy = (r2 - r0)/delta;
    M22f dfdad(drdx.x, drdy.x,
                drdx.y, drdy.y);
    V2f d_ad = dfdad.inv()*df;
    AffineMap& thisAff = editPreTrans ? preMap : postMap;
    thisAff.m.a *= (1+d_ad.x);
    thisAff.m.c *= (1+d_ad.x);
    thisAff.m.b *= (1+d_ad.y);
    thisAff.m.d *= (1+d_ad.y);
}

void FlameMapping::rotate(V2f p, V2f df, bool editPreTrans)
{
    FlameMapping tmpMap = *this;
    AffineMap& aff = editPreTrans ? tmpMap.preMap : tmpMap.postMap;
    const float delta = 0.001;
    V2f r0 = tmpMap.map(p);
    aff.m = M22f( cos(delta), sin(delta),
            -sin(delta), cos(delta)) * aff.m;
    V2f r1 = tmpMap.map(p);
    V2f dfdtheta = (r1 - r0)/delta;
    float dtheta = dot(df, dfdtheta) / dot(dfdtheta, dfdtheta);
    AffineMap& thisAff = editPreTrans ? preMap : postMap;
    thisAff.m = M22f( cos(dtheta), sin(dtheta),
                        -sin(dtheta), cos(dtheta)) * thisAff.m;
}



//------------------------------------------------------------------------------
void computeFractalFlame(PointVBO* points, const FlameMaps& flameMaps)
{
    IFSPoint* ptData = points->mapBuffer(GL_WRITE_ONLY);

    int nMaps = flameMaps.maps.size();
    // Flame fractals!
    V2f p = V2f(0,0);
    int discard = 20;
    C3f col(1);
    int nPoints = points->size();
    for(int i = -discard; i < nPoints; ++i)
    {
        int mapIdx = rand() % nMaps;
        const FlameMapping& m = flameMaps.maps[mapIdx];
        p = m.map(p);
        col = 0.5*(col + m.col);
        if(i >= 0)
        {
            ptData->pos = p;
            ptData->col = col;
            ++ptData;
        }
    }
    points->unmapBuffer();
}



//------------------------------------------------------------------------------

#if 0
    // Julia set example.
    complex z = 0;
    complex c(0.4,0.3);
    C3f c1(1,0,0);
    C3f c2(0,1,0);
    C3f col(0);
    int batchSize = points->size();
    for(int i = 0; i < batchSize; ++i, ++ptData)
    {
        if(float(rand())/RAND_MAX > 0.5)
        {
            z = sqrt(z - c);
            col = 0.5f*(col + c1);
        }
        else
        {
            z = -sqrt(z - c);
            col = 0.5f*(col + c2);
        }
        ptData->pos = V2f(real(z), imag(z));
        ptData->col = col;
    }

    // Set of Halton points
    for(int i = 0; i < (int)points->size(); ++i, ++ptData)
    {
        ptData->pos = V2f(2*radicalInverse<2>(i) - 1, 2*radicalInverse<3>(i) - 1);
        heatmap(double(i)/points->size(), ptData->col.x, ptData->col.y, ptData->col.z);
    }
#endif

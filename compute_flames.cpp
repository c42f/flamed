#include "compute_flames.h"

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

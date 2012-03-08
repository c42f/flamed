#define COMPILE_FOR_GPU
#include "compute_flames.h"

#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#define ASSERT_CUDA_SUCCESS(expr)                                              \
    do {                                                                       \
        cudaError_t err = expr;                                                \
        if(err != cudaSuccess)                                                 \
        {                                                                      \
            std::cerr << "Cuda error: \"" << cudaGetErrorString(err)           \
                << "\" in calling " #expr "\n";                                \
            abort();                                                           \
        }                                                                      \
    } while(false)


__global__ void flameGenKernel(IFSPoint* points, size_t nPoints)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= nPoints)
        return;
    points[i].pos.x = 2.0f*i/nPoints - 1;
    points[i].pos.y = 2.0f*(radicalInverse<2>(i)) - 1;
    points[i].col.x = 1;
    points[i].col.y = 0;
    points[i].col.z = 0;
}

void initCuda()
{
    ASSERT_CUDA_SUCCESS(cudaGLSetGLDevice(0));
}

void computeFractalFlameGPU(PointVBO* points, const FlameMaps& flameMaps)
{
    // Can't get the new API to work for some reason...
//    cudaGraphicsResource_t *cudaRes = 0;
//    ASSERT_CUDA_SUCCESS(cudaGraphicsGLRegisterBuffer(cudaRes, points->id(),
//                                    cudaGraphicsRegisterFlagsWriteDiscard));
//    ASSERT_CUDA_SUCCESS(cudaGraphicsMapResources(1, cudaRes));
    ASSERT_CUDA_SUCCESS(cudaGLRegisterBufferObject(points->id()));

    IFSPoint* ifsPoints = 0;
    ASSERT_CUDA_SUCCESS(cudaGLMapBufferObject((void**)&ifsPoints, points->id()));
//    size_t nBytes = 0;
//    ASSERT_CUDA_SUCCESS(cudaGraphicsResourceGetMappedPointer((void**)&ifsPoints,
                                                             //&nBytes, *cudaRes));
    assert(nBytes/sizeof(IFSPoint) == points->size());

    const size_t blockSize = 128;
    dim3 block(blockSize, 1, 1);
    dim3 grid(ceildiv(points->size(), blockSize), 1, 1);
    flameGenKernel<<<grid, block>>>(ifsPoints, points->size());

    ASSERT_CUDA_SUCCESS(cudaGLUnmapBufferObject(points->id()));
    ASSERT_CUDA_SUCCESS(cudaGLUnregisterBufferObject(points->id()));
//    ASSERT_CUDA_SUCCESS(cudaGraphicsUnmapResources(1, cudaRes));
//    ASSERT_CUDA_SUCCESS(cudaGraphicsUnregisterResource(*cudaRes));


    /*
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
        col = m.colorSpeed*m.col + (1-m.colorSpeed)*col;
        if(i >= 0)
        {
            ptData->pos = p;
            ptData->col = col;
            ++ptData;
        }
    }
    */
}


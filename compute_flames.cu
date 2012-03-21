#define COMPILE_FOR_GPU
#include "compute_flames.h"

#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

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

#define ASSERT_KERNEL_SUCCESS(str)                                             \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if(err != cudaSuccess)                                                 \
        {                                                                      \
            std::cerr << "Cuda error: \"" << cudaGetErrorString(err)           \
                << "\" in calling " str "\n";                                  \
            abort();                                                           \
        }                                                                      \
    } while(false)


// TODO: Fix this awful hard coded maximum!
#define MAX_MAPS 20

void initCuda()
{
    ASSERT_CUDA_SUCCESS(cudaGLSetGLDevice(0));
}


__global__ void rngInitKernel(curandState_t* generators, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    curand_init(42, i, i, &generators[i]);
}


__global__ void flameGenKernel(IFSPoint* points, curandState_t* rngs,
                               int nThreads, long long nPoints,
                               FlameMapping* flameMaps, int nMaps)
{
    long long id = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ FlameMapping maps[MAX_MAPS];
    if(threadIdx.x < nMaps+1)
        maps[threadIdx.x] = flameMaps[threadIdx.x];
    syncthreads();
    curandState_t gen = rngs[id];
    const int discard = 20;
    V2f p(0);
    C3f col(0);
    for(int i = 0; i < discard; ++i)
    {
        float rnd = curand_uniform(&gen);
        int mapIdx = int(rnd*nMaps);
        const FlameMapping& m = maps[mapIdx];
        p = m.map(p);
        col = m.colorSpeed*m.col + (1-m.colorSpeed)*col;
    }
    for(long long i = id; i < nPoints; i += nThreads)
    {
        float rnd = curand_uniform(&gen);
        int mapIdx = int(rnd*nMaps);
        const FlameMapping& m = maps[mapIdx];
        p = m.map(p);
        col = m.colorSpeed*m.col + (1-m.colorSpeed)*col;
        points[i].pos = maps[nMaps].map(p); // final map is last one
        points[i].col = col;
    }
    rngs[id] = gen;
}


struct GPUFlameEngine::Pimpl
{
    int nThreads;
    thrust::device_vector<FlameMapping> flameMaps;
    thrust::device_vector<curandState_t> randState;

    Pimpl(int nThreads)
        : nThreads(nThreads),
        flameMaps(),
        randState(nThreads)
    { }
};


GPUFlameEngine::GPUFlameEngine()
    : m_pimpl(new Pimpl(40000))
{
    const int blockSize = 256;
    rngInitKernel<<<ceildiv(m_pimpl->nThreads,blockSize), blockSize>>>(
        thrust::raw_pointer_cast(&m_pimpl->randState[0]), m_pimpl->nThreads);
    ASSERT_KERNEL_SUCCESS("rngInitKernel");
}


void GPUFlameEngine::generate(PointVBO* points, const FlameMaps& flameMaps)
{
    // Can't get the new API to work for some reason...
//    cudaGraphicsResource_t *cudaRes = 0;
//    ASSERT_CUDA_SUCCESS(cudaGraphicsGLRegisterBuffer(cudaRes, points->id(),
//                                    cudaGraphicsRegisterFlagsWriteDiscard));
//    ASSERT_CUDA_SUCCESS(cudaGraphicsMapResources(1, cudaRes));
//    size_t nBytes = 0;
//    ASSERT_CUDA_SUCCESS(cudaGraphicsResourceGetMappedPointer((void**)&ifsPoints,
//                                                             &nBytes, *cudaRes));
//    assert(nBytes/sizeof(IFSPoint) == points->size());
//    ASSERT_CUDA_SUCCESS(cudaGraphicsUnmapResources(1, cudaRes));
//    ASSERT_CUDA_SUCCESS(cudaGraphicsUnregisterResource(*cudaRes));

    ASSERT_CUDA_SUCCESS(cudaGLRegisterBufferObject(points->id()));

    thrust::device_vector<FlameMapping>& flameMaps_d = m_pimpl->flameMaps;
    flameMaps_d = flameMaps.maps;
    flameMaps_d.push_back(flameMaps.finalMap);
    if(flameMaps_d.size() > MAX_MAPS)
        flameMaps_d.resize(MAX_MAPS);

    IFSPoint* ifsPoints = 0;
    ASSERT_CUDA_SUCCESS(cudaGLMapBufferObject((void**)&ifsPoints, points->id()));

    const int blockSize = 256;
    flameGenKernel<<<ceildiv(m_pimpl->nThreads, blockSize), blockSize>>>(
        ifsPoints, thrust::raw_pointer_cast(&m_pimpl->randState[0]),
        m_pimpl->nThreads, points->size(),
        thrust::raw_pointer_cast(&flameMaps_d[0]),
        flameMaps.maps.size()
    );
    ASSERT_KERNEL_SUCCESS("flameGenKernel");

    ASSERT_CUDA_SUCCESS(cudaGLUnmapBufferObject(points->id()));
    ASSERT_CUDA_SUCCESS(cudaGLUnregisterBufferObject(points->id()));
}


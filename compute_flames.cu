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
        int mapIdx = curand(&gen) % nMaps;
        const FlameMapping& m = maps[mapIdx];
        p = m.map(p);
        col = m.colorSpeed*m.col + (1-m.colorSpeed)*col;
    }
    for(long long i = id; i < nPoints; i += nThreads)
    {
        int mapIdx = curand(&gen) % nMaps;
        const FlameMapping& m = maps[mapIdx];
        p = m.map(p);
        col = m.colorSpeed*m.col + (1-m.colorSpeed)*col;
        // "out of loop" map is last one in the maps array.
        points[i].pos = maps[nMaps].map(p);
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
    : m_pimpl()
{
    const int blockSize = 256;
    const int nThreads = blockSize*(40000/blockSize);
    m_pimpl.reset(new Pimpl(nThreads));
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


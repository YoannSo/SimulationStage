/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "render_particles.h"
#include "render_mesh.h"
 // Particle system class
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
    ~ParticleSystem();

    enum ParticleConfig
    {
        CONFIG_RANDOM,
        CONFIG_GRID,
        CONFIG_TOP,
        _NUM_CONFIGS
    };

    enum ParticleArray
    {
        POSITION,
        VELOCITY,
        TRIANGLES,
    };

    void update(float deltaTime);
    void reset(ParticleConfig config);
    void launchForce();
    float* getArray(ParticleArray array);
    void   setArray(ParticleArray array, const float* data, int start, int count);

    int    getNumParticles() const
    {
        return m_numParticles;
    }

    unsigned int getCurrentReadBuffer() const
    {
        return m_posVbo;
    }
    unsigned int getCurrentReadBufferTriangle() const
    {
        return m_triangleVBO;
    }
    unsigned int getColorBuffer()       const
    {
        return m_colorVBO;
    }
    float getCubeSize() {
        return m_params.sizeCubeZ;
    }




    void* getCudaPosVBO()              const
    {
        return (void*)m_cudaPosVBO;
    }
    void* getCudaColorVBO()            const
    {
        return (void*)m_cudaColorVBO;
    }
    void setCubeSizeX(float x)
    {
        m_params.sizeCubeX = x;
    }
    void setCubeSizeZ(float x)
    {
        m_params.sizeCubeZ = x;
    }


    void dumpGrid();
    void dumpParticles(uint start, uint count);
    void setRenderer(ParticleRenderer* myRenderer) {
        this->myRenderer = myRenderer;
    }
    void setMeshRenderer(MeshRenderer* myRenderer) {
        this->meshRenderer = myRenderer;
    }
    void setIterations(int i)
    {
        m_solverIterations = i;
    }

    void setDamping(float x)
    {
        m_params.globalDamping = x;
    }
    void setGravity(float x)
    {
        m_params.gravity = make_float3(0.0f, x, 0.0f);
    }
    void setTakeScreen(bool b) {
        this->takeScreen = b;
    }
    void setCollideSpring(float x)
    {
        m_params.spring = x;
    }
    void setCollideDamping(float x)
    {
        m_params.damping = x;
    }
    void setMaxCycles(int x) {
        m_params.maxNbCycles = x;
    }
    void setInclinaison(float x) {
        m_params.inclinaison = x;
    }
    void setPumpForce(float x) {
        m_params.pumpForce = x;
    }
    void setCollideShear(float x)
    {
        m_params.shear = x;
    }
    void setCollideAttraction(float x)
    {
        m_params.attraction = x;
    }

    void setColliderPos(float3 x)
    {
        m_params.colliderPos = x;
    }
    void setTrianglePos(float3 p0,float3 p1,float3 p2,float3 p3)
    {
        m_params.p0 = p0;
        m_params.p1 = p1;
        m_params.p2 = p2;
        m_params.p3 = p3;
       
    }
    void addAllTriangles(int nbTriangles, float3* triangles) {
        m_params.nbTrianglesPoints = nbTriangles;
        int nbTriangle=nbTriangles;
        cudaMalloc((void**)&nbTriangle, sizeof(int));

        cudaMemcpyToSymbol("nbT", &nbTriangle, sizeof(int), 0, cudaMemcpyHostToDevice);
       
    }
    float3* getTrianglesPoints() {
        return m_params.trianglesPoints;
    }
    int getNbTrianglePoints() {
        return m_params.nbTrianglesPoints;
    }
    float3 getP0()
    {
        return m_params.p0;
    }
    float3 getP1()
    {
        return m_params.p1;
    }
    float3 getP2()
    {
        return m_params.p2;
    }
    float3 getP3()
    {
        return m_params.p3;
    }
    float getParticleRadius()
    {
        return m_params.particleRadius;
    }
    float3 getColliderPos()
    {
        return m_params.colliderPos;
    }
    float getColliderRadius()
    {
        return m_params.colliderRadius;
    }
    uint3 getGridSize()
    {
        return m_params.gridSize;
    }
    float3 getWorldOrigin()
    {
        return m_params.worldOrigin;
    }
    float3 getCellSize()
    {
        return m_params.cellSize;
    }

    void addSphere(int index, float* pos, float* vel, int r, float spacing);

protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();

    void initGrid(uint* size, float spacing, float jitter, uint numParticles);
    void initGridTop(uint* size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint m_numParticles;

    // CPU data
    float* m_hPos;              // particle positions
    float* m_hVel;              // particle velocities
    float* m_hTriangle;
    uint* m_hParticleHash;
    uint* m_hCellStart;
    uint* m_hCellEnd;

    // GPU data
    float* m_dPos;
    float* m_dVel;
    float* m_dTriangle;

    float* m_dSortedPos;
    float* m_dSortedVel;



    // grid data for sorting method
    uint* m_dGridParticleHash; // grid hash value for each particle
    uint* m_dGridParticleIndex;// particle index for each particle
    uint* m_dCellStart;        // index of start of each cell in sorted list
    uint* m_dCellEnd;          // index of end of cell

    uint   m_gridSortBits;

    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors
    uint m_triangleVBO;

    float* m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float* m_cudaColorVBO;      // these are the CUDA deviceMem Color
    float* m_cudaTriangleVBO;

    struct cudaGraphicsResource* m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource* m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource* m_cuda_trianglevbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint m_numGridCells;

    StopWatchInterface* m_timer;

    uint m_solverIterations;

    ParticleRenderer* myRenderer;
    MeshRenderer* meshRenderer;
    int idScreenshot = 0;
    bool takeScreen = false;
};

#endif // __PARTICLESYSTEM_H__

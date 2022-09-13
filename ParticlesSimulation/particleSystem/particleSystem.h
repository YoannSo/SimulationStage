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
#include "../renderer/render_particles.h"
#include "../renderer/render_mesh.h"
<<<<<<< HEAD
#include "../renderer/render_CopperBalls.h"

=======
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
#include "../mesh/triangle_mesh_model.h"

 // Particle system class
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL,std::string meshName,float p_pourcentageCopperBall);
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
        TRIANGLECPY,
        INDICES,
        COPPERCPY
    };

    enum GridType {
        TRIANGLE,
        COPPER
    };
    void update(float deltaTime);
    void reset(ParticleConfig config);
    int3 calcGridPos(float3 p);
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
    int getNumCopperBalls() {
        return nbTriangleSelected;
    }
    float* getCopperBallsVertexBuffer() {
        float* vertexBuffer = new float[nbTriangleSelected*4];
        for (int i = 0; i < nbTriangleSelected; i++) {
            vertexBuffer[i * 4] = m_hBalls[i * 4];
            vertexBuffer[i * 4 +1] = m_hBalls[i * 4 +1];
            vertexBuffer[i * 4 +2] = m_hBalls[i * 4 +2];
            vertexBuffer[i * 4 +3] = 1.f;

        }
        return vertexBuffer;
    }

    void dumpGrid();
    void dumpGridTriangle();
    void dumpParticles(uint start, uint count);
    void setRenderer(ParticleRenderer* myRenderer) {
        this->myRenderer = myRenderer;
    }
    void setMeshRenderer(MeshRenderer* myRenderer) {
        this->_meshRenderer = myRenderer;
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
        m_params.trianglesPoints = (float3 *) malloc(3 * sizeof(float3) * nbTriangles);
        for (int i = 0; i < nbTriangles * 3; i++) {
            printf("%d", i);
            m_params.trianglesPoints[i] = triangles[i];
        }
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
        return m_params.particuleCellSize;
    }
    MeshRenderer* getMeshRenderer() { return _meshRenderer; }
    uint calcGridHash(int3 gridPos);
    void setTriangleGrid();
    void decomposerTriangleRec(float3 p0, float3 p1, float3 p2, int nbSub,int idTriangle);
    void addIndicesInGrid(int3 gridPos,uint hash, int indiceTriangle, GridType type);
    void addSphere(int index, float* pos, float* vel, int r, float spacing);
    void loadMesh(const std::string& p_name, const std::string& p_path);
    void copyToAdaptedVector(int debut, int fin, int nbTriangle);
    void reset();
    void setAllTriangleBuffers(float x, float y, float z);
<<<<<<< HEAD
    void setAllCopperBallBuffers(float transX, float transY, float transZ);
    void clearBuffers(float X,float Y, float Z);
    void getCopperBallResult();
    float* getCopperBallPos();
    float* getCopperBallsColor();
=======
    void test(float X,float Y, float Z);
    void getCopperBallResult();
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();
    void setCopperBallGrid();
    void initGrid(uint* size, float spacing, float jitter, uint numParticles);
    void initGridTop(uint* size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;

    float m_pourcentageCopperBall;
    uint m_numParticles;
    // CPU data
    float* m_hPos;              // particle positions
    float* m_hVel;              // particle velocities
    unsigned int* m_hIndices;
    float* m_hTriangles;

    uint* m_hParticleHash;
    uint* m_hCellStart;
    uint* m_hCellEnd;
    
    // GPU data
    float* m_dPos;
    float* m_dVel;
    unsigned int* m_dIndices;
    float* m_dTriangles;


    float* m_dSortedPos;
    float* m_dSortedVel;
    float* m_dSortedTriangles;


    // grid data for sorting method
    uint* m_dGridParticleHash; // grid hash value for each particle
    uint* m_dGridParticleIndex;// particle index for each particle

<<<<<<< HEAD
    int* m_hGridTrianglesIndex;// particle index for each particle

    int* m_dGridTrianglesHash;// particle index for each particle
    int* m_hGridTrianglesHash;// particle index for each particle

    std::vector<int> m_hIdTriangleInSimulation = std::vector<int>();
    int nbTriangleSelected;
    float* m_hBalls;

    std::vector<int> m_hGridTrianglesAdaptedIndex= std::vector<int>();// particle index for each particle
    std::vector<int> m_hGridCopperBallsAdaptedIndex = std::vector<int>();// particle index for each particle

    float* m_hCopperBall;
    int* m_dGridCopperBalls;
    int* m_hGridCopperBallsHash;
    int* m_dGridCopperBallsHash;
    int* m_hGridCopperBalls;
    uint _maxCopperBallsPerCell = 40;
=======
    int* m_dGridTrianglesIndex;// particle index for each particle
    int* m_hGridTrianglesIndex;// particle index for each particle

    std::vector<int> m_hGridTrianglesAdaptedIndex= std::vector<int>();// particle index for each particle

    std::vector<float> m_hCopperBalls = std::vector<float>();// particle index for each particle
    int* m_dGridCopperBalls;
    int* m_hGridCopperBalls;
    uint _maxCopperBallsPerCell=2;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    uint* m_dNumInteraction;
    uint* m_hNumInteraction;

    float* m_dCopperBalls;

    int* m_dGridTrianglesAdaptedIndex;// particle index for each particle
<<<<<<< HEAD
    std::string m_meshName;
  
=======

    int* m_dGridTrianglesHash;// particle index for each particle
    int* m_hGridTrianglesHash;// particle index for each particle
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

   // uint* m_dGridTrianglesHash;// particle index for each particle

    uint* m_dCellStart;        // index of start of each cell in sorted list
    uint* m_dCellEnd;          // index of end of cell

    uint* m_dCellStartTriangle;        // index of start of each cell in sorted list
    uint* m_dCellEndTriangle;          // index of end of cell

    uint   m_gridSortBits;

    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors

    float* m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float* m_cudaColorVBO;      // these are the CUDA deviceMem Color
<<<<<<< HEAD

    float* m_cudaColorTriangleVBO;
    struct cudaGraphicsResource* m_cudeposvbo_resource;
    uint m_colorTriangleVBO;

=======
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    struct cudaGraphicsResource* m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource* m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint m_numGridCells;

    uint3 m_gridTriangleSize; // how many cells for each axes
    uint m_numTriangleGridCells; // how many cell in total
    float3 m_triangleGridWorldOrigin; // center of the grid
    float3 m_triangleGridWorldSize; // grid size
    float m_cellSize;
    StopWatchInterface* m_timer;

    uint m_solverIterations;

    ParticleRenderer* myRenderer;
    MeshRenderer* _meshRenderer;
    CopperBallsRenderer* m_copperBallsRenderer;
    int _nbVertices = 0;
    int _nbTriangles = 0;
    int idScreenshot = 0;
    int _sizeIndices=0;
    int _sizeTriangles = 0;
    int _maxTrianglePerBox = 100;
    int _nbTriangleSubMax = 3;

    bool takeScreen = false;
};

#endif // __PARTICLESYSTEM_H__

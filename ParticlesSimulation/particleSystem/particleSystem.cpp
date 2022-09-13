<<<<<<< HEAD
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

// OpenGL Graphics includes
=======
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

 // OpenGL Graphics includes
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#pragma comment(lib, "glew32.lib")
#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <iostream>
#include <string>
int compteurCycle = 0;
<<<<<<< HEAD
ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL, std::string p_meshName,float p_pourcentageCopperBall) :
=======
ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_hTriangles(0),
    m_hIndices(0),
    m_dPos(0),
    m_dVel(0),
    m_dIndices(0),
    m_dTriangles(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
<<<<<<< HEAD
    float3 worldSize = make_float3(2.0f, 2.0f, 2.f);
=======
        float3 worldSize = make_float3(2.0f, 2.0f, 2.f);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

<<<<<<< HEAD
    m_params.particleRadius = 1.0f / 192.0f;
=======
    m_params.particleRadius = 1.0f /192.0f; // TODO: la particule du coin qui passe => chelou ?
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    m_params.colliderPos = make_float3(0.f, 0.f, 0.f);
    m_params.p0 = make_float3(0.f, 0.f, 0.f);
    m_params.p1 = make_float3(0.f, 0.f, 0.f);
    m_params.p2 = make_float3(0.f, 0.f, 0.f);
<<<<<<< HEAD
=======
    m_params.numberOfBalls = 0;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    m_params.colliderRadius = 0.2f;
    m_params.nbCycles = 0;
    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    m_params.triangleCellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.particuleCellSize = make_float3(cellSize, cellSize, cellSize);
<<<<<<< HEAD
=======
    printf("%f %f %f nbGrille \n", m_params.triangleCellSize.x, m_params.triangleCellSize.y, m_params.triangleCellSize.z);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;
    m_params.maxBallPerCell = _maxCopperBallsPerCell;
    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;
<<<<<<< HEAD
    m_meshName = p_meshName;
    m_pourcentageCopperBall = p_pourcentageCopperBall;
=======

>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    _initialize(numParticles);
    m_params.nbVertices = _nbVertices;
    m_params.maxTrianglePerbox = _maxTrianglePerBox;

}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

<<<<<<< HEAD
float* ParticleSystem::getCopperBallsColor()
{
    float* color = new float[nbTriangleSelected * 4];
    int max = -1;
    int somme = 0;
    int courant;
    int id = 0;
    for (int i = 0; i < nbTriangleSelected; i++) {
        courant = m_hNumInteraction[i];
        somme += courant;
        if (courant > max) {
            id = i;
            max = courant;

        }
    }
    float moy = (float)somme / nbTriangleSelected;
    int numberOfSuperior = 0;
    for (int i = 0; i < nbTriangleSelected; i++) {

        if (m_hNumInteraction[i] > moy) {
            color[i * 4] = 1.f;
            color[i * 4 + 1] = 0.f;
            color[i * 4 + 2] = 0.f;
            numberOfSuperior++;

        }
        else {
            color[i * 4] = 0.f;
            color[i * 4 + 1] = 0.f;
            color[i * 4 + 2] = 1.f;
        }
        color[i * 4 + 3] = 1.f;
    }
    return color;
}

=======
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// create a color ramp
void colorRamp(float t, float* r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors - 1);
    int i = (int)t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i + 1][0], u);
    r[1] = lerp(c[i][1], c[i + 1][1], u);
    r[2] = lerp(c[i][2], c[i + 1][2], u);
<<<<<<< HEAD

}
=======
 
}
// TODO : les grilles on l'aire bien remplie ? Il faut ensuite ajouter un test, pour tester les boules de cuivre si on toche on incremente un tab, et on met -1 dans un autre tab pour dire cettre particule a deja interagit avec cette boule
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;
<<<<<<< HEAD

    _meshRenderer = new MeshRenderer("Ceramique", "../data/objets/"+m_meshName);

    _sizeIndices = _meshRenderer->getEboSize();
    _sizeTriangles = (int)_meshRenderer->getEboSize() / 3;
=======
    
    _meshRenderer = new MeshRenderer("test", "../data/copper_balls/menger.obj");
   //_meshRenderer = new MeshRenderer("test", "../data/avion_papier/avion_papier.obj");
    std::ifstream monFlux("../data/copper_balls/menger_d10.txt");
    if (monFlux) {
        std::string ligne; //Une variable pour stocker les lignes lues
        std::getline(monFlux, ligne); // enlever premiere ligne
        float number;
        while (monFlux >> ligne) //on lit mot par mot
        {
           m_hCopperBalls.emplace_back(std::stof(ligne));
           m_params.numberOfBalls++;
        }
    }
    else
    {
        std::cout << "ERREUR: Impossible d'ouvrir le fichier en lecture." << std::endl;
    }
    m_params.numberOfBalls = m_params.numberOfBalls / 4;
    _sizeIndices = _meshRenderer->getEboSize();
    _sizeTriangles = (int)_meshRenderer->getEboSize()/3;
    printf(" j'ai: %d \n", m_params.numberOfBalls);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    // allocate host storage
    m_hPos = new float[m_numParticles * 4];
    m_hVel = new float[m_numParticles * 4];
    m_hIndices = new unsigned int[_sizeIndices];

    TriangleMeshModel model = _meshRenderer->getModel();


    int nbCellX, nbCellZ;
    float3 min = model.getAABBMin();
    float3 max = model.getAABBMax();
    m_triangleGridWorldSize = make_float3(max.x - min.x, max.y - min.y, max.z - min.z);
<<<<<<< HEAD
    m_triangleGridWorldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    m_cellSize = m_triangleGridWorldSize.y / m_gridSize.y;
=======
    //m_triangleGridWorldOrigin = make_float3((max.x + min.x) * 0.5f, (max.y + min.y) * 0.5f, (max.z + min.z) * 0.5f);
    m_triangleGridWorldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
     m_cellSize = m_triangleGridWorldSize.y / m_gridSize.y;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    nbCellX = 1 + m_triangleGridWorldSize.x / m_cellSize;
    nbCellZ = 1 + m_triangleGridWorldSize.z / m_cellSize;

    m_gridTriangleSize = make_uint3(nbCellX, m_gridSize.y, nbCellZ);
    m_numTriangleGridCells = m_gridTriangleSize.x * m_gridTriangleSize.y * m_gridTriangleSize.z;
<<<<<<< HEAD

 
    m_params.numberOfBalls = nbTriangleSelected;
=======
    printf("Je dois afficher %dx%dx%d cellules soit :%d cellules\n", m_gridTriangleSize.x, m_gridTriangleSize.y, m_gridTriangleSize.z, m_numTriangleGridCells);
    printf("Ma grille fait x:%f y:%f z:%f  et son centre est: x:%f y:%f z:%f \n", m_triangleGridWorldSize.x, m_triangleGridWorldSize.y, m_triangleGridWorldSize.z, m_triangleGridWorldOrigin.x, m_triangleGridWorldOrigin.y, m_triangleGridWorldOrigin.z);


>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    _nbTriangles = model._nbTriangles;
    _nbVertices = model._nbVertices;
    m_hTriangles = new float[_nbVertices * 4];
    m_hGridTrianglesIndex = new int[m_numGridCells * _maxTrianglePerBox];
    m_hGridCopperBalls = new int[m_numGridCells * _maxCopperBallsPerCell];
<<<<<<< HEAD
    m_hGridTrianglesHash = new int[m_numGridCells * 2];
    m_hGridCopperBallsHash = new int[m_numGridCells * 2];



    m_params.nbIndices = _sizeIndices;

    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));
   

    memset(m_hIndices, 0, _sizeIndices * sizeof(unsigned int));
    memset(m_hTriangles, 0, _nbVertices * 4 * sizeof(float));
=======
    m_hGridTrianglesHash = new int[m_numGridCells];
    m_hNumInteraction = new uint[m_params.numberOfBalls];


    m_params.nbIndices = _sizeIndices;
   // _maxTrianglePerBox = _nbTriangles;

    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));

    memset(m_hIndices, 0, _sizeIndices  * sizeof(unsigned int));
    memset(m_hTriangles, 0, _nbVertices * 4 * sizeof(float));
    printf("num: %d max %d \n", m_numGridCells, _maxTrianglePerBox);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    memset(m_hGridTrianglesIndex, -1, m_numGridCells * sizeof(int) * _maxTrianglePerBox);
    memset(m_hGridCopperBalls, -1, m_numGridCells * sizeof(int) * _maxCopperBallsPerCell);

<<<<<<< HEAD
    memset(m_hGridTrianglesHash, -1, m_numGridCells * 2 * sizeof(int));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));


    memset(m_hGridCopperBallsHash, -1, m_numGridCells * 2 * sizeof(int));


    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;
    unsigned int testMem = sizeof(float) * 4 * _nbVertices;
    unsigned int indicesMem = sizeof(unsigned int) * _sizeIndices;
=======
    memset(m_hGridTrianglesHash, -1, m_numGridCells * sizeof(int));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));
        
    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

    memset(m_hNumInteraction, 0, m_params.numberOfBalls * sizeof(uint));
    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;
    unsigned int testMem= sizeof(float) * 4 * _nbVertices;
    unsigned int indicesMem = sizeof(unsigned int)  * _sizeIndices;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);


    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&m_cudaPosVBO, memSize));

    }

    allocateArray((void**)&m_dVel, memSize);
    allocateArray((void**)&m_dTriangles, testMem);

    allocateArray((void**)&m_dIndices, indicesMem);

    allocateArray((void**)&m_dSortedPos, memSize);
    allocateArray((void**)&m_dSortedVel, memSize);

<<<<<<< HEAD


    allocateArray((void**)&m_dGridParticleHash, m_numParticles * sizeof(uint));

    allocateArray((void**)&m_dGridParticleIndex, m_numParticles * sizeof(uint));

    allocateArray((void**)&m_dGridTrianglesHash, m_numGridCells * 2 * sizeof(int));
=======
    allocateArray((void**)&m_dCopperBalls, m_params.numberOfBalls*4*sizeof(float));


    allocateArray((void**)&m_dGridParticleHash, m_numParticles * sizeof(uint));
   // allocateArray((void**)&m_dGridTrianglesHash, _sizeIndices * sizeof(uint)*3);

    allocateArray((void**)&m_dGridParticleIndex, m_numParticles * sizeof(uint));
    allocateArray((void**)&m_dGridTrianglesIndex, m_numGridCells*_maxTrianglePerBox*sizeof(int));
    allocateArray((void**)&m_dGridCopperBalls, m_numGridCells* _maxCopperBallsPerCell * sizeof(int));

    allocateArray((void**)&m_dGridTrianglesHash, m_numGridCells * sizeof(int));
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    allocateArray((void**)&m_dCellStart, m_numGridCells * sizeof(uint));
    allocateArray((void**)&m_dCellEnd, m_numGridCells * sizeof(uint));

    allocateArray((void**)&m_dCellStartTriangle, m_numGridCells * sizeof(uint));
    allocateArray((void**)&m_dCellEndTriangle, m_numGridCells * sizeof(uint));

<<<<<<< HEAD

    allocateArray((void**)&m_dGridCopperBallsHash, m_numGridCells * 2 * sizeof(int));
=======
    allocateArray((void**)&m_dNumInteraction, m_params.numberOfBalls * sizeof(uint));
    setArray(COPPERCPY, m_hCopperBalls.data(), 0, m_params.numberOfBalls * 4);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    if (m_bUseOpenGL)
    {
        m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
        float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float* ptr = data;

        for (uint i = 0; i < m_numParticles; i++)
        {
            float t = i / (float)m_numParticles;
#if 0
            * ptr++ = rand() / (float)RAND_MAX;
            *ptr++ = rand() / (float)RAND_MAX;
            *ptr++ = rand() / (float)RAND_MAX;
#else
            colorRamp(t, ptr);
            ptr += 3;
#endif
            * ptr++ = 1.0f;
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&m_cudaColorVBO, sizeof(float) * numParticles * 4));
    }

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);
<<<<<<< HEAD
=======
    setCopperBallGrid();
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);
    delete[] m_hTriangles;
    delete[] m_hPos;
    delete[] m_hVel;
    delete[] m_hCellStart;
    delete[]m_hGridCopperBalls;
    delete[] m_hCellEnd;
    delete[] m_hIndices;
    delete[] m_hGridTrianglesIndex;
    delete[] m_hGridTrianglesHash;
    delete[] m_hNumInteraction;
<<<<<<< HEAD
    delete[] m_hGridCopperBallsHash;
    delete[] m_hCopperBall;
    m_hGridTrianglesAdaptedIndex.clear();
    m_hGridCopperBallsAdaptedIndex.clear();
    freeArray(m_dGridCopperBallsHash);
    freeArray(m_dGridTrianglesAdaptedIndex);
=======
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    freeArray(m_dNumInteraction);
    freeArray(m_dCopperBalls);
    freeArray(m_dIndices);
    freeArray(m_dVel);
    freeArray(m_dTriangles);
    freeArray(m_dSortedPos);
    freeArray(m_dGridTrianglesHash);
    freeArray(m_dSortedVel);
    freeArray(m_dGridCopperBalls);
    freeArray(m_dGridParticleHash);
<<<<<<< HEAD
=======
    freeArray(m_dGridTrianglesIndex);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
    freeArray(m_dCellStartTriangle);
    freeArray(m_dCellEndTriangle);

    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_colorvbo_resource);
        unregisterGLBufferObject(m_cuda_posvbo_resource);

        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);

    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));

    }
}
void ParticleSystem::launchForce() {
    launchFlowForce(m_dVel,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells, this->m_params.inclinaison, this->m_params.pumpForce);
}
// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float* dPos;
    float* dTriangles;
    if (m_bUseOpenGL)
    {
        dPos = (float*)mapGLBufferObject(&m_cuda_posvbo_resource);

    }
    else
    {
        dPos = (float*)m_cudaPosVBO;

    }
<<<<<<< HEAD

=======
   
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    // update constants
    setParameters(&m_params);

    // integrate
<<<<<<< HEAD
    integrateSystem(
        dPos,
        m_dVel,
        m_dTriangles,
        m_dIndices,
        m_dCopperBalls,
        deltaTime,
        m_numParticles,
        m_dGridCopperBalls,
        m_dGridCopperBallsHash,
        m_dNumInteraction,

        m_dGridTrianglesHash,
        m_dGridTrianglesAdaptedIndex

    );
=======
        integrateSystem(
            dPos,
            m_dVel,
            m_dTriangles,
            m_dIndices,
            m_dCopperBalls,
        deltaTime,
        m_numParticles,
            m_dGridTrianglesIndex,
            m_dGridCopperBalls,
            m_dNumInteraction
            );
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_dVel,
        m_numParticles,
        m_numGridCells);

    // process collisions

    collide(
        m_dVel,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells, this->m_params.inclinaison);
<<<<<<< HEAD
    if (this->takeScreen && this->idScreenshot < 5 && this->m_params.nbCycles == 20 && compteurCycle % 2 == 0) {
=======
    if (this->takeScreen && this->idScreenshot < 5 && this->m_params.nbCycles == 20&& compteurCycle%2==0) {
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        this->idScreenshot++;
        compteurCycle = 0;
    }
    if (this->m_params.nbCycles > this->m_params.maxNbCycles) {
        this->m_params.nbCycles = 0;
        compteurCycle++;

        launchFlowForce(m_dVel,
            m_dSortedPos,
            m_dSortedVel,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            m_numParticles,
            m_numGridCells, this->m_params.inclinaison, this->m_params.pumpForce);
<<<<<<< HEAD

    }

=======
        
    }
    
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    this->m_params.nbCycles++;
    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
<<<<<<< HEAD
=======
      //  unmapGLBufferObject(m_cuda_trianglevbo_resource);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    }
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint) * m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint) * m_numGridCells);
    uint maxCellSize = 0;

    for (uint i = 0; i < m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}
void
ParticleSystem::dumpGridTriangle()
{
<<<<<<< HEAD

=======
    // dump grid information
   // copyArrayFromDevice(m_hCellStart, m_dGridTrianglesHash, 0, sizeof(uint) * m_numGridCells);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint) * m_numGridCells);
    uint maxCellSize = 0;

    for (uint i = 0; i < m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}
void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float) * 4 * count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);

    for (uint i = start; i < start + 1; i++)
    {
        printf("pos: (%.4f)\n", m_hPos[i * 4 + 1]);

    }
}

float*
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float* hdata = 0;
    float* ddata = 0;
    struct cudaGraphicsResource* cuda_vbo_resource = 0;

    switch (array)
    {
    default:
    case POSITION:
        hdata = m_hPos;
        ddata = m_dPos;
        cuda_vbo_resource = m_cuda_posvbo_resource;
        break;

    case VELOCITY:
        hdata = m_hVel;
        ddata = m_dVel;
        break;
<<<<<<< HEAD

=======
  
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float* data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
    default:
    case POSITION:
    {
        if (m_bUseOpenGL)
        {
            unregisterGLBufferObject(m_cuda_posvbo_resource);
            glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
            glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
        }
        else
        {
            copyArrayToDevice(m_cudaPosVBO, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        }
    }
    break;

    case VELOCITY:
        copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        break;
    case TRIANGLECPY:
<<<<<<< HEAD
        copyArrayToDevice(m_dTriangles, data, start * 4 * sizeof(float), _nbVertices * 4 * sizeof(float));
        break;
    case INDICES:
        copyArrayToDevice(m_dIndices, data, start * sizeof(unsigned int), _sizeIndices * sizeof(unsigned int));
        break;
    case COPPERCPY:
        copyArrayToDevice(m_dCopperBalls, data, start * sizeof(float), count * sizeof(float));
=======
        //TriangleMeshModel currentModel = meshRenderer->getModel();
       // int nbPoints = currentModel._nbVertices;
      //  if (m_bUseOpenGL)
        //{
          //  unregisterGLBufferObject(m_cuda_trianglevbo_resource);
            //glBindBuffer(GL_ARRAY_BUFFER, m_triangleVBO);
            //glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), _nbVertices * 4 * sizeof(float), data);
            //glBindBuffer(GL_ARRAY_BUFFER, 0);
            //registerGLBufferObject(m_triangleVBO, &m_cuda_trianglevbo_resource);
        //}
        //else
        //{
            copyArrayToDevice(m_dTriangles, data, start * 4 * sizeof(float), _nbVertices * 4 * sizeof(float));
       // }
        //copyArrayToDevice(m_dTestTriangle, data, start * sizeof(float), _nbVertices * sizeof(float));
        break;
    case INDICES:
        copyArrayToDevice(m_dIndices, data, start  * sizeof(unsigned int), _sizeIndices * sizeof(unsigned int));
        break;
    case COPPERCPY:
        copyArrayToDevice(m_dCopperBalls, data, start * sizeof(float),count*sizeof(float));
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        break;
    }
}

inline float frand()
{
    return rand() / (float)RAND_MAX;
}

void
ParticleSystem::initGrid(uint* size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z = 0; z < size[2]; z++)
    {
        for (uint y = 0; y < size[1]; y++)
        {
            for (uint x = 0; x < size[0]; x++)
            {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i * 4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 3] = 1.0f;

                    m_hVel[i * 4] = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;
                }
            }
        }
    }
}
void
ParticleSystem::initGridTop(uint* size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

<<<<<<< HEAD
    for (uint z = 0; z < size[2]; z++)

    {
        for (uint y = 0; y < size[1]; y++)
=======
                for (uint z = 0; z < size[2]; z++)

    {
                    for (uint y = 0; y < size[1]; y++)
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

        {
            for (uint x = 0; x < size[0]; x++)
            {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;
                if (i < numParticles)
                {
                    m_hPos[i * 4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
<<<<<<< HEAD
                    m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius + 1.5f - spacing + (frand() * 2.0f - 1.0f) * jitter;
=======
                    m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius +1.5f-spacing + (frand() * 2.0f - 1.0f) * jitter;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
                    m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 3] = 1.0f;

                    m_hVel[i * 4] = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;
                }
            }
        }
    }
}

<<<<<<< HEAD
void ParticleSystem::setAllTriangleBuffers(float transX, float transY, float transZ) {

    TriangleMeshModel currentModel = _meshRenderer->getModel();
    int j = 0;
    for (int i = 0; i < _nbVertices * 4; i += 4) {
        m_hTriangles[i] = currentModel._meshes[0]._vertices[j]._position.x + transX;
        m_hTriangles[i + 1] = currentModel._meshes[0]._vertices[j]._position.y + transY;
        m_hTriangles[i + 2] = currentModel._meshes[0]._vertices[j]._position.z + transZ;
=======
void ParticleSystem::setAllTriangleBuffers(float transX,float transY,float transZ) {

    TriangleMeshModel currentModel = _meshRenderer->getModel();
    //int nbTriangles = currentModel._nbTriangles;
    //int nbPoints = currentModel._nbVertices;
    printf("%d %d \n", _nbVertices, _nbTriangles);
    int j = 0;
    for (int i = 0; i < _nbVertices * 4; i += 4) {
        m_hTriangles[i] = currentModel._meshes[0]._vertices[j]._position.x+ transX;
        m_hTriangles[i + 1] = currentModel._meshes[0]._vertices[j]._position.y+ transY;
        m_hTriangles[i + 2] = currentModel._meshes[0]._vertices[j]._position.z+ transZ;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        m_hTriangles[i + 3] = 1.f;
        j++;
    }
    for (int i = 0; i < _sizeIndices; i++) {
        m_hIndices[i] = currentModel._meshes[0]._indices[i];
    }


    setArray(TRIANGLECPY, m_hTriangles, 0, _nbVertices);
    copyArrayToDevice(m_dIndices, m_hIndices, 0 * sizeof(unsigned int), _sizeIndices * sizeof(unsigned int));
<<<<<<< HEAD

    setTriangleGrid();

    m_hIdTriangleInSimulation.shrink_to_fit();
   

}
void ParticleSystem::setAllCopperBallBuffers(float transX, float transY, float transZ) {

    setCopperBallGrid();
=======
    // calcTriangleGrid(m_dGridTrianglesIndex, m_dTriangles,
      //   _nbTriangles);
    setTriangleGrid();
    //sortTriangles(m_dGridTrianglesHash, m_dGridTrianglesIndex, _sizeIndices);
   /*reorderDataAndFindCellStartTriangle(
        m_dCellStartTriangle,
        m_dCellEndTriangle,
        m_dSortedTriangles,
        m_dGridTrianglesHash,
        m_dGridTrianglesIndex,
        m_dTriangles,
        _sizeIndices,
        m_numGridCells);*/
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
}
void ParticleSystem::reset()
{
    delete[] m_hTriangles;
    delete[] m_hGridTrianglesIndex;
    delete[] m_hGridTrianglesHash;
<<<<<<< HEAD

    m_hIdTriangleInSimulation.clear();
    m_hGridTrianglesAdaptedIndex.clear();
    freeArray(m_dGridTrianglesAdaptedIndex);
    freeArray(m_dGridTrianglesHash);

    freeArray(m_dTriangles);
=======
    delete[]m_hGridCopperBalls;
    freeArray(m_dGridCopperBalls);
    freeArray(m_dTriangles);
    freeArray(m_dGridTrianglesHash);
    freeArray(m_dGridTrianglesIndex);
    m_hGridCopperBalls = new int[m_numGridCells * _maxCopperBallsPerCell];
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60



    m_hTriangles = new float[_nbVertices * 4];
    m_hGridTrianglesIndex = new int[m_numGridCells * _maxTrianglePerBox];
<<<<<<< HEAD
    m_hGridTrianglesHash = new int[m_numGridCells * 2];

    memset(m_hGridTrianglesIndex, -1, m_numGridCells * sizeof(int) * _maxTrianglePerBox);
    memset(m_hGridTrianglesHash, -1, m_numGridCells * 2 * sizeof(int));
    memset(m_hTriangles, 0, _nbVertices * 4 * sizeof(float));


    allocateArray((void**)&m_dGridTrianglesHash, m_numGridCells * 2 * sizeof(int));
    allocateArray((void**)&m_dTriangles, sizeof(float) * 4 * _nbVertices);

}
void ParticleSystem::clearBuffers(float x, float y, float z)
{
    delete[] m_hBalls;
    delete[]m_hGridCopperBalls;
    delete[] m_hGridCopperBallsHash;
    freeArray(m_dGridCopperBallsHash);
    freeArray(m_dGridCopperBalls);
    freeArray(m_dCopperBalls);
    m_hGridCopperBalls = new int[m_numGridCells * _maxCopperBallsPerCell];
    m_hGridCopperBallsHash = new int[m_numGridCells * 2];
    memset(m_hGridCopperBalls, -1, m_numGridCells * sizeof(int) * _maxCopperBallsPerCell);
    memset(m_hGridCopperBallsHash, -1, m_numGridCells * sizeof(int) * 2);
    allocateArray((void**)&m_dGridCopperBallsHash, m_numGridCells * 2 * sizeof(int));


    setAllTriangleBuffers(x, y, z);

    m_hIdTriangleInSimulation.shrink_to_fit();
    nbTriangleSelected = m_hIdTriangleInSimulation.size() * m_pourcentageCopperBall;
    printf("Dans l'espace de Simulation il y a : %d triangles. On va prendre %d billes de cuibres", m_hIdTriangleInSimulation.size(), nbTriangleSelected);
    m_hNumInteraction = new uint[nbTriangleSelected];
    m_hCopperBall = new float[nbTriangleSelected * 4];
    m_hBalls = new float[nbTriangleSelected * 4];
    memset(m_hCopperBall, 0, nbTriangleSelected * 4 * sizeof(float));
    memset(m_hBalls, 0, nbTriangleSelected * 4 * sizeof(float));
    memset(m_hNumInteraction, 0, nbTriangleSelected * sizeof(uint));
    allocateArray((void**)&m_dCopperBalls, nbTriangleSelected * 4 * sizeof(float));



    for (int i = 0; i < nbTriangleSelected; i++) {
        int idRand = (rand() ^ (rand() << 15)) % m_hIdTriangleInSimulation.size();
        int idTri = m_hIdTriangleInSimulation[idRand];
        int idP0 = m_hIndices[idTri] * 4;
        int idP1 = m_hIndices[idTri + 1] * 4;
        int idP2 = m_hIndices[idTri + 2] * 4;

        float3 p0 = make_float3(m_hTriangles[idP0], m_hTriangles[idP0 + 1], m_hTriangles[idP0 + 2]);
        float3 p1 = make_float3(m_hTriangles[idP1], m_hTriangles[idP1 + 1], m_hTriangles[idP1 + 2]);
        float3 p2 = make_float3(m_hTriangles[idP2], m_hTriangles[idP2 + 1], m_hTriangles[idP2 + 2]);

        float3 baryCentre = make_float3((p0.x + p1.x + p2.x) / 3.f, (p0.y + p1.y + p2.y) / 3.f, (p0.z + p1.z + p2.z) / 3.f);

        m_hBalls[i * 4] = baryCentre.x;
        m_hBalls[i * 4 + 1] = baryCentre.y;
        m_hBalls[i * 4 + 2] = baryCentre.z;
        float radius = 0.003f + (float)(rand() % (int)(RAND_MAX * 0.009)) / RAND_MAX;
        m_hBalls[i * 4 + 3] = radius;
    }


    copyArrayToDevice(m_dCopperBalls, m_hBalls, 0, nbTriangleSelected * 4 * sizeof(float));

    setAllCopperBallBuffers(x, y, z);
=======
    m_hGridTrianglesHash = new int[m_numGridCells];


    memset(m_hGridTrianglesIndex, -1, m_numGridCells * sizeof(int) * _maxTrianglePerBox);
    memset(m_hGridTrianglesHash, -1, m_numGridCells * sizeof(int));
    memset(m_hTriangles, 0, _nbVertices * 4 * sizeof(float));
    memset(m_hGridCopperBalls, -1, m_numGridCells * sizeof(int) * _maxCopperBallsPerCell);

    allocateArray((void**)&m_dGridTrianglesIndex, m_numGridCells * _maxTrianglePerBox * sizeof(int));
    allocateArray((void**)&m_dGridCopperBalls, m_numGridCells * _maxCopperBallsPerCell * sizeof(int));

    allocateArray((void**)&m_dGridTrianglesHash, m_numGridCells * sizeof(int));
    allocateArray((void**)&m_dTriangles, sizeof(float) * 4 * _nbVertices);
}
void ParticleSystem::test(float x,float y,float z)
{
    setAllTriangleBuffers(x,y,z);

   // printf(" %f %f %f \n", m_hTriangles[0]+x, m_hTriangles[1]+y,m_hTriangles[2]+z);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
}
void
ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
    default:
    case CONFIG_RANDOM:
    {
        int p = 0, v = 0;

        for (uint i = 0; i < m_numParticles; i++)
        {
            float point[3];
            point[0] = frand();
            point[1] = frand();
            point[2] = frand();
            m_hPos[p++] = 2 * (point[0] - 0.5f);
            m_hPos[p++] = 2 * (point[1] - 0.5f);
            m_hPos[p++] = 2 * (point[2] - 0.5f);
            m_hPos[p++] = 1.0f; // radius
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
        }
    }
    break;

    case CONFIG_GRID:
    {
        float jitter = m_params.particleRadius * 0.01f;
        uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
        uint gridSize[3];
        gridSize[0] = gridSize[1] = gridSize[2] = s;
        initGrid(gridSize, m_params.particleRadius * 2, jitter, m_numParticles);
    }
    break;

    case CONFIG_TOP: {
        float jitter = m_params.particleRadius * 0.01f;
        float spacing = m_params.particleRadius * 4.f;
        int nbParticlesMaxPerLine = (int)2.f / (spacing);
        uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
        uint gridSize[3];
        gridSize[0] = nbParticlesMaxPerLine;
        int nbProf = nbParticlesMaxPerLine;
        gridSize[2] = nbParticlesMaxPerLine;
<<<<<<< HEAD
        gridSize[1] =   m_numParticles / (nbProf * nbParticlesMaxPerLine) + 1;//100.f
        initGridTop(gridSize, spacing, jitter, m_numParticles);


    }
                   break;
=======
        printf("prof %d %d %d\n", nbProf, nbParticlesMaxPerLine);
        gridSize[1] =  60.f; // m_numParticles / (nbProf * nbParticlesMaxPerLine) + 1;
        initGridTop(gridSize, spacing, jitter, m_numParticles);

        
    }
    break;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    }
    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);

}
// calculate position in uniform grid
int3 ParticleSystem::calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - m_params.worldOrigin.x) / m_params.triangleCellSize.x);
    gridPos.y = floorf((p.y - m_params.worldOrigin.y) / m_params.triangleCellSize.y);
    gridPos.z = floorf((p.z - m_params.worldOrigin.z) / m_params.triangleCellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
uint ParticleSystem::calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (m_params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (m_params.gridSize.y - 1);
    gridPos.z = gridPos.z & (m_params.gridSize.z - 1);
<<<<<<< HEAD
    return ((gridPos.z * m_params.gridSize.y) * m_params.gridSize.x) + (gridPos.y * m_params.gridSize.x) + gridPos.x;
}
void ParticleSystem::copyToAdaptedVector(int debut, int fin, int nbTriangle) {

=======
    return ((gridPos.z * m_params.gridSize.y)* m_params.gridSize.x) + (gridPos.y* m_params.gridSize.x) + gridPos.x;
}
void ParticleSystem::copyToAdaptedVector( int debut, int fin,int nbTriangle) {
    
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    m_hGridTrianglesAdaptedIndex.emplace_back(nbTriangle);
    for (int i = debut; i < fin; i++) {
        m_hGridTrianglesAdaptedIndex.emplace_back(m_hGridTrianglesIndex[i]);
    }
}
<<<<<<< HEAD
void ParticleSystem::getCopperBallResult() {
    copyArrayFromDevice(m_hNumInteraction, m_dNumInteraction, 0, nbTriangleSelected * sizeof(uint));
    for (int i =0 ; i < nbTriangleSelected; i++) {
    }
    getCopperBallsColor();
}
float* ParticleSystem::getCopperBallPos() {
    return m_hBalls;
}

void ParticleSystem::setCopperBallGrid() {


    for (int i = 0; i < nbTriangleSelected; i++) {
        float3 pos = make_float3(m_hBalls[i * 4], m_hBalls[i * 4 + 1], m_hBalls[i * 4 + 2]);
=======
void ParticleSystem::getCopperBallResult () {
    copyArrayFromDevice(m_hNumInteraction, m_dNumInteraction,0, m_params.numberOfBalls * sizeof(uint));
    for (int i =0 ; i < m_params.numberOfBalls; i++) {
        printf("Je suis la boule %d et i get hit %d times \n", i, m_hNumInteraction[i]);
    }
}
void ParticleSystem::setCopperBallGrid() {
  
    for (int i = 0; i < m_params.numberOfBalls;i++) {
        float3 pos = make_float3(m_hCopperBalls[i*4], m_hCopperBalls[i *4  + 1], m_hCopperBalls[i*4 + 2]);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        int3 gridPos = calcGridPos(pos);
        uint hash = calcGridHash(gridPos);
        addIndicesInGrid(gridPos, hash, i, COPPER);

    }
<<<<<<< HEAD

    int nbCourant = 0;
    int idVector = 0;
    int idVectorCourant = 0;
    int idGrid = 0;
    int sizeOfTab = _maxCopperBallsPerCell * m_numGridCells;

    for (int i = 0; i < sizeOfTab; i++) {
        nbCourant++;
        if (nbCourant == 1 && m_hGridCopperBalls[i] == -1) {
            i += _maxCopperBallsPerCell - nbCourant;
            m_hGridCopperBallsHash[idGrid] = -1;
            m_hGridCopperBallsHash[idGrid + 1] = 0;
            idGrid += 2;
            nbCourant = 0;
            continue;
        }
        if (m_hGridCopperBalls[i] == -1 || nbCourant == _maxCopperBallsPerCell) {
            i += _maxCopperBallsPerCell - nbCourant;
            m_hGridCopperBallsHash[idGrid] = idVectorCourant;
            m_hGridCopperBallsHash[idGrid + 1] = nbCourant - 1;
            idVectorCourant += idVector;
            idVector = 0;
            nbCourant = 0;
            idGrid += 2;
            continue;
        }
        m_hGridCopperBallsAdaptedIndex.emplace_back(m_hGridCopperBalls[i]);
        idVector++;
    }
    m_hGridCopperBallsAdaptedIndex.shrink_to_fit();
    allocateArray((void**)&m_dGridCopperBalls, m_hGridCopperBallsAdaptedIndex.size() * sizeof(int));

    allocateArray((void**)&m_dNumInteraction, nbTriangleSelected * sizeof(uint));

    copyArrayToDevice(m_dGridCopperBalls, m_hGridCopperBallsAdaptedIndex.data(), 0, m_hGridCopperBallsAdaptedIndex.size() * sizeof(int));

    copyArrayToDevice(m_dGridCopperBallsHash, m_hGridCopperBallsHash, 0, m_numGridCells * 2 * sizeof(int));


}
void ParticleSystem::setTriangleGrid() {
    int sizeOfTab = _maxTrianglePerBox * m_numGridCells;
    for (int i = 0; i < _sizeIndices; i += 3) {
        int idP0 = m_hIndices[i] * 4;
        int idP1 = m_hIndices[i + 1] * 4;
        int idP2 = m_hIndices[i + 2] * 4;
=======
    for (int i = 0; i < m_numGridCells * _maxCopperBallsPerCell; i++) {
        //printf(" dans %d j'ai %d \n", i, m_hGridCopperBalls[i]);
    }
    copyArrayToDevice(m_dGridCopperBalls, m_hGridCopperBalls, 0, m_numGridCells * _maxCopperBallsPerCell * sizeof(int));

    
}
void ParticleSystem::setTriangleGrid() {
    int sizeOfTab = _maxTrianglePerBox * m_numGridCells;
    printf("%d \n", _sizeIndices);
    for (int i = 0; i < _sizeIndices; i+=3) {
        int idP0 =m_hIndices[i]*4;
        int idP1 = m_hIndices[i+1]*4;
        int idP2 = m_hIndices[i+2]*4;
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

        float3 p0 = make_float3(m_hTriangles[idP0], m_hTriangles[idP0 + 1], m_hTriangles[idP0 + 2]);
        float3 p1 = make_float3(m_hTriangles[idP1], m_hTriangles[idP1 + 1], m_hTriangles[idP1 + 2]);
        float3 p2 = make_float3(m_hTriangles[idP2], m_hTriangles[idP2 + 1], m_hTriangles[idP2 + 2]);

        int3 gridPos1 = calcGridPos(p0);
        int3 gridPos2 = calcGridPos(p1);
        int3 gridPos3 = calcGridPos(p2);


        uint hash1 = calcGridHash(gridPos1);
        uint hash2 = calcGridHash(gridPos2);
        uint hash3 = calcGridHash(gridPos3);
<<<<<<< HEAD

        decomposerTriangleRec(p0, p1, p2, 0, i);


    }

    int nbCourant = 0;
    int idVector = 0;
    int idVectorCourant = 0;
    int idGrid = 0;
    for (int i = 0; i < sizeOfTab; i++) {
        nbCourant++;
        if (nbCourant == 1 && m_hGridTrianglesIndex[i] == -1) {
            i += _maxTrianglePerBox - nbCourant;
            m_hGridTrianglesHash[idGrid] = -1;
            m_hGridTrianglesHash[idGrid + 1] = 0;
            idGrid += 2;
            nbCourant = 0;
            continue;

        }
        if (m_hGridTrianglesIndex[i] == -1 || nbCourant == _maxTrianglePerBox) {
            i += _maxTrianglePerBox - nbCourant;
            m_hGridTrianglesHash[idGrid] = idVectorCourant;
            m_hGridTrianglesHash[idGrid + 1] = nbCourant - 1;
            idVectorCourant += idVector;
            idVector = 0;
            nbCourant = 0;
            idGrid += 2;
            continue;
        }
        m_hGridTrianglesAdaptedIndex.emplace_back(m_hGridTrianglesIndex[i]);
        idVector++;
    }



    m_hGridTrianglesAdaptedIndex.shrink_to_fit();

    copyArrayToDevice(m_dGridTrianglesHash, m_hGridTrianglesHash, 0, m_numGridCells * 2 * sizeof(int));




    allocateArray((void**)&m_dGridTrianglesAdaptedIndex, m_hGridTrianglesAdaptedIndex.size() * sizeof(int));


    copyArrayToDevice(m_dGridTrianglesAdaptedIndex, m_hGridTrianglesAdaptedIndex.data(), 0, m_hGridTrianglesAdaptedIndex.size() * sizeof(int));
   

}
void ParticleSystem::decomposerTriangleRec(float3 p0, float3 p1, float3 p2, int nbSub, int idTriangle) {

=======
      //  printf("id:%d   -- %d %d %d \n", i, gridPos1.x, gridPos1.y, gridPos1.z);

        // printf("je met %d dans %d %d et %d \n",i, hash1, hash2, hash3);
       // printf(" %d %d %d -- %d %d %d -- %d %d %d -- %d %d %d\n", gridPos1.x, gridPos1.y, gridPos1.z, gridPos2.x, gridPos2.y, gridPos2.z, gridPos3.x, gridPos3.y, gridPos3.z,hash1,hash2,hash3);
       decomposerTriangleRec(p0, p1, p2, 0, i);
        /*bool needHash2 = false;
        bool needHash3 = false;
        if (hash1 != hash2)
            needHash2 = true;
        if (hash3 != hash1 && hash3 != hash2)
            needHash3 = true;
        for (int j = hash1 * _maxTrianglePerBox; j < hash1 * _maxTrianglePerBox + _maxTrianglePerBox; j++) {
            if (m_hGridTrianglesIndex[j] == -1) {
                m_hGridTrianglesIndex[j] = i;
                break;
            }
        }
        if (needHash2) {
            for (int j = hash2 * _maxTrianglePerBox; j < hash2 * _maxTrianglePerBox + _maxTrianglePerBox; j++) {
                if (m_hGridTrianglesIndex[j] == -1) {
                    m_hGridTrianglesIndex[j] = i;
                    break;
                }
            }
        }
        if (needHash3) {
            for (int j = hash3 * _maxTrianglePerBox; j < hash3 * _maxTrianglePerBox + _maxTrianglePerBox; j++) {
                if (m_hGridTrianglesIndex[j] == -1) {
                    m_hGridTrianglesIndex[j] = i;
                    break;
                }
            }
        }*/
       
   }
   /* int indiceInHash = 0;
    int* test;
    int debut; // c/c de debut a fin dans un nouveau tableu 
    int nbTriangles;
    for (int i = 0; i < m_numGridCells; i++) {
        int debut = i;
        int fin = i;
        int compteurDeTriangle = 0;
        for (int j = i; j < sizeOfTab; j++) {
            if (m_hGridTrianglesIndex[j] == -1 || compteurDeTriangle == _maxTrianglePerBox ) {
                fin = j-1;
                break;
            }
            compteurDeTriangle++;
        }
        if (fin < debut) {
           m_hGridTrianglesHash[i] = -1;
        }
        else {
            m_hGridTrianglesHash[i]= m_hGridTrianglesAdaptedIndex.size();
        }
        nbTriangles = fin - debut +1;
        // c/c debut a fin
        copyToAdaptedVector(debut, fin, nbTriangles);
        // je dois avoir indice triangle, un compteur de valeur. Je met le compteur dans index puis tous les indices de triangle. et dans hash l'indice de la case
    }
    for (int i = 0; i < m_numGridCells; i++) {
        printf(" la case : %d a: %d \n", i, m_hGridTrianglesHash[i]);
        int k = m_hGridTrianglesHash[i];
        if (k != -1) {
            int nb = m_hGridTrianglesAdaptedIndex.at(k);

        }
       // for (int j = k; j < nb + k; j++) {
          //  printf(" j'ai en indice: %d\n",m_hGridTrianglesAdaptedIndex[j]);

        //}
    }*/
    copyArrayToDevice(m_dGridTrianglesIndex, m_hGridTrianglesIndex, 0 , sizeOfTab * sizeof(int));

}
void ParticleSystem::decomposerTriangleRec(float3 p0, float3 p1, float3 p2, int nbSub,int idTriangle) {
    
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    int3 gridPos1 = calcGridPos(p0);
    int3 gridPos2 = calcGridPos(p1);
    int3 gridPos3 = calcGridPos(p2);
    uint hash1 = calcGridHash(gridPos1);
    uint hash2 = calcGridHash(gridPos2);
    uint hash3 = calcGridHash(gridPos3);

    if (hash1 == hash2 && hash1 == hash3)
    {
<<<<<<< HEAD
        addIndicesInGrid(gridPos1, hash1, idTriangle, TRIANGLE);
=======
        // printf("p0: %d %d %d p1: %d %d %d p2: %d %d %d \n", gridPos1.x, gridPos1.y, gridPos1.z, gridPos2.x, gridPos2.y, gridPos2.z, gridPos3.x, gridPos3.y, gridPos3.z);
        addIndicesInGrid(gridPos1,hash1, idTriangle,TRIANGLE);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        return;

    }


    if (nbSub == _nbTriangleSubMax)
    {
<<<<<<< HEAD

        addIndicesInGrid(gridPos1, hash1, idTriangle, TRIANGLE);
        addIndicesInGrid(gridPos2, hash2, idTriangle, TRIANGLE);
        addIndicesInGrid(gridPos3, hash3, idTriangle, TRIANGLE);
        return;
    }


    float3 baryCentre = make_float3((p0.x + p1.x + p2.x) / 3.f, (p0.y + p1.y + p2.y) / 3.f, (p0.z + p1.z + p2.z) / 3.f);
    if (hash1 != hash2) {
        decomposerTriangleRec(p0, p1, baryCentre, nbSub + 1, idTriangle);
    }
    if (hash1 != hash3) {
        decomposerTriangleRec(p0, p2, baryCentre, nbSub + 1, idTriangle);

    }
    if (hash2 != hash3) {
        decomposerTriangleRec(p1, p2, baryCentre, nbSub + 1, idTriangle);
=======
        
        /*if (hash1 == hash2 && hash1 == hash3)
        {
            addIndicesInGrid(hash1, idTriangle);
        }
        else if (hash1 != hash2 && hash1 != hash3 && hash2 == hash3) {
            addIndicesInGrid(hash1, idTriangle);
            addIndicesInGrid(hash2, idTriangle);
        }*/
      //  else if (hash1 != hash2 && hash1 != hash3 && hash2 != hash3) {
            addIndicesInGrid(gridPos1,hash1, idTriangle,TRIANGLE);
            addIndicesInGrid(gridPos2,hash2, idTriangle, TRIANGLE);
            addIndicesInGrid(gridPos3,hash3, idTriangle, TRIANGLE);

     //   }
       /* else if (hash1 != hash3)
        {
            addIndicesInGrid(hash1, idTriangle);
            addIndicesInGrid(hash3, idTriangle);

        }
        else if (hash1 != hash2)
        {
            addIndicesInGrid(hash1, idTriangle);
            addIndicesInGrid(hash2, idTriangle);

        }*/
        return;

    }
 
  
    float3 baryCentre = make_float3((p0.x + p1.x + p2.x) / 3.f, (p0.y + p1.y + p2.y) / 3.f, (p0.z + p1.z + p2.z) / 3.f);
    if (hash1 != hash2) {
        decomposerTriangleRec(p0, p1, baryCentre, nbSub+1,idTriangle);
    }
    if (hash1 != hash3) {
        decomposerTriangleRec(p0, p2, baryCentre, nbSub+1, idTriangle);

    }
    if (hash2 != hash3) {
        decomposerTriangleRec(p1, p2, baryCentre, nbSub+1, idTriangle);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    }
}

<<<<<<< HEAD
void ParticleSystem::addIndicesInGrid(int3 gridPos, uint hash, int id, GridType type) {
    if (gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0 || gridPos.x>128 || gridPos.y>128 || gridPos.z>128) {
=======
void ParticleSystem::addIndicesInGrid(int3 gridPos,uint hash,int id,GridType type) {
    if (gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0 || gridPos.x>128|| gridPos.y>128|| gridPos.z>128) {
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        return;
    }
    switch (type) {

    case TRIANGLE:
        for (int j = hash * _maxTrianglePerBox; j < hash * _maxTrianglePerBox + _maxTrianglePerBox; j++) {
            if (m_hGridTrianglesIndex[j] == id) {
                return;
            }
            if (m_hGridTrianglesIndex[j] == -1) {
                m_hGridTrianglesIndex[j] = id;
<<<<<<< HEAD
                m_hIdTriangleInSimulation.emplace_back(id);
=======
                // printf("je met %d dans %d  \n", idTriangle, hash);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
                return;
            }
        }
        break;
    case COPPER:
        for (int j = hash * _maxCopperBallsPerCell; j < hash * _maxCopperBallsPerCell + _maxCopperBallsPerCell; j++) {
<<<<<<< HEAD
            if (m_hGridCopperBalls[j] == -1) {
                m_hGridCopperBalls[j] = id;
=======
            if (m_hGridCopperBalls[j] == id) {
                return;
            }
            if (m_hGridCopperBalls[j] == -1) {
                m_hGridCopperBalls[j] = id;
                printf("je met %d dans %d  \n", id, hash);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
                return;
            }
        }
        break;
    }
<<<<<<< HEAD
=======
   
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

}
void
ParticleSystem::addSphere(int start, float* pos, float* vel, int r, float spacing)
{
    uint index = start;

    for (int z = -r; z <= r; z++)
    {
        for (int y = -r; y <= r; y++)
        {
            for (int x = -r; x <= r; x++)
            {
                float dx = x * spacing;
                float dy = y * spacing;
                float dz = z * spacing;
                float l = sqrtf(dx * dx + dy * dy + dz * dz);
                float jitter = m_params.particleRadius * 0.01f;

                if ((l <= m_params.particleRadius * 2.0f * r) && (index < m_numParticles))
                {
                    m_hPos[index * 4] = pos[0] + dx + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 1] = pos[1] + dy + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 2] = pos[2] + dz + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 3] = pos[3];

                    m_hVel[index * 4] = vel[0];
                    m_hVel[index * 4 + 1] = vel[1];
                    m_hVel[index * 4 + 2] = vel[2];
                    m_hVel[index * 4 + 3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}

void ParticleSystem::loadMesh(const std::string& p_name, const std::string& p_path)
{

}

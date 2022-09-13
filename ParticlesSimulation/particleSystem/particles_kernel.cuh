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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;


    float3 p0;
    float3 p1;
    float3 p2;
    float3 p3;
    int nbIndices;
    int maxTrianglePerbox;
    float3* trianglesPoints;
    int nbTrianglesPoints;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint3 gridTriangleSize;

    uint numCells;
    float3 worldOrigin;
    float3 triangleCellSize;
    float3 particuleCellSize;
    int maxBallPerCell;
    int numberOfBalls;
    uint numBodies;
    uint maxParticlesPerCell;
    float nbVertices;
    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
    int nbCycles;
    int maxNbCycles;
    float inclinaison;
    float pumpForce;
    float3 flowForce;
    float sizeCubeX;
    float sizeCubeZ;
};

#endif

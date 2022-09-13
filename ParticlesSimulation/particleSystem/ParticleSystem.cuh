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

extern "C"
{
    void cudaInit(int argc, char** argv);

    void allocateArray(void** devPtr, int size);
    void freeArray(void* devPtr);

    void threadSync();

    void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource** cuda_vbo_resource, int size);
    void copyArrayToDevice(void* device, const void* host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);
    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);


    void setParameters(SimParams* hostParams);

    void integrateSystem(float* pos,
        float* vel,
        float* triangles,
        unsigned int* indices,
        float* copperBalls,
        float deltaTime,
<<<<<<< HEAD
        uint numParticles,int * gridCopperBalls,int* gridCopperBallHash,uint* numInteraction,int* gridHashTriangle,int* gridTriangleAdapted);
=======
        uint numParticles, int* gridTrianglesIndex,int * gridCopperBalls,uint* numInteraction);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60

    void calcHash(uint* gridParticleHash,
        uint* gridParticleIndex,
        float* pos,
        int    numParticles);
    void calcTriangleGrid(
        uint* gridTrianglesIndex,
        float* posTriangles,
        int    numTriangles);
    void reorderDataAndFindCellStart(uint* cellStart,
        uint* cellEnd,
        float* sortedPos,
        float* sortedVel,
        uint* gridParticleHash,
        uint* gridParticleIndex,
        float* oldPos,
        float* oldVel,
        uint   numParticles,
        uint   numCells);

    void collide(float* newVel,
        float* sortedPos,
        float* sortedVel,
        uint* gridParticleIndex,
        uint* cellStart,
        uint* cellEnd,
        uint   numParticles,
        uint   numCells, float inclinaison);
    void launchFlowForce(float* newVel,
        float* sortedPos,
        float* sortedVel,
        uint* gridParticleIndex,
        uint* cellStart,
        uint* cellEnd,
        uint   numParticles,
        uint   numCells,
        float inclinaison,
        float pumpForce);
    void sortParticles(uint* dGridParticleHash, uint* dGridParticleIndex, uint numParticles);
<<<<<<< HEAD

=======
    void reorderDataAndFindCellStartTriangle(uint* cellStart,
        uint* cellEnd,
        float* sortedTriangle,
        uint* gridTriangleHash,
        uint* gridTriangleIndex,
        float* oldTriangle,
        uint   numTriangles,
        uint   numCells);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
    void sortTriangles(uint* dGridTriangleHash, uint* dGridTriangleIndex, uint numTriangles);
}

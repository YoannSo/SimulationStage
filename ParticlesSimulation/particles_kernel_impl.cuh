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

 /*
  * CUDA particle system kernel code.
  */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;


struct integrate_functor
{
    float deltaTime;
    float* triangles;
    unsigned int* indices;
    __host__ __device__
        integrate_functor(float delta_time,float* p_triangles,unsigned int* p_indices) : deltaTime(delta_time), triangles(p_triangles),indices(p_indices) {}

    template <typename Tuple>
    __device__
        void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
       vel += params.gravity * deltaTime;
        vel *= params.globalDamping;
        
       // new position = old position + velocity * deltaTime
       for (int i = 0; i < params.nbIndices; i += 3) {
           int idP0 = indices[i]*4;
           int idP1 = indices[i + 1] * 4;
           int idP2 = indices[i+2]*4;
          // printf("%d %d %d %d \n", i, idP0, idP1, idP2);
            float3 p0 = make_float3(triangles[idP0], triangles[idP0 + 1], triangles[idP0 + 2]);
            float3 p1 = make_float3(triangles[idP1 ], triangles[idP1 + 1], triangles[idP1 + 2]);
            float3 p2 = make_float3(triangles[idP2 ], triangles[idP2 + 1], triangles[idP2 + 2]);
            float triangleCollider = collideTriangle(pos, vel, p0, p1, p2);
            //printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n", p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);

           if (triangleCollider != 0)
            {

               float3 dir= normalize(vel);
               float3 pointInTriangle = pos + dir * triangleCollider;
                float firstMembre = pointInTriangle.x - pos.x;
                float secondMembre = pointInTriangle.y - pos.y;

                float thirdmember = pointInTriangle.z - pos.z;
                float dist=sqrtf(firstMembre*firstMembre+secondMembre*secondMembre+thirdmember*thirdmember);
                float3 v0v1 = p1 - p0;
                float3 v0v2 = p2 - p0;
                // no need to normalize
                //float3 N = cross(v0v2,v0v1);  //N 
                //N = normalize(N);
               // float theta = dot(N, pos - pointInTriangle);
                //if (theta < 0)
                  //  N = -N; 
              //  float3 N = cross(v0v2, v0v1);  //N 
               // N = normalize(N);
              //  float3 lol = pointInTriangle - pos;
                //printf("%f %f %f %f %f %f %f %f %f \n", vel.x, vel.y, vel.z, lol.x, lol.y, lol.z);
                float tailleQuiPasse = params.particleRadius - dist;
                float3 N = cross(v0v1,v0v2 );  //N 
                N = normalize(N);

                float3 reflectDir = dir - 2.f * dot(N, dir) * N;

                vel *= params.boundaryDamping;
                vel = make_float3(0.f);
                vel = reflectDir * tailleQuiPasse*1.5f;

                //vel += tailleQuiPasse * reflectDir;

            }
        }
        
        pos += vel * deltaTime;
        float sizeCubex = params.sizeCubeX;
        float sizeCubey = 1.f;
        float sizeCubez = 0.65f;
        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > sizeCubex - params.particleRadius)
        {
            pos.x = sizeCubex - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -sizeCubex + params.particleRadius)
        {
            pos.x = -sizeCubex + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > sizeCubey+30.f - params.particleRadius)
        {
            pos.y = sizeCubey - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > params.sizeCubeZ - params.particleRadius)
        {
            pos.z = params.sizeCubeZ - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }



#endif

        if (pos.y < -sizeCubey + params.particleRadius)
        {
            pos.y = -sizeCubey + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floorf((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floorf((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y - 1);
    gridPos.z = gridPos.z & (params.gridSize.z - 1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint* gridParticleHash,  // output
    uint* gridParticleIndex, // output
    float4* pos,               // input: positions
    uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint* cellStart,        // output: cell start index
    uint* cellEnd,          // output: cell end index
    float4* sortedPos,        // output: sorted positions
    float4* sortedVel,        // output: sorted velocities
    uint* gridParticleHash, // input: sorted grid hashes
    uint* gridParticleIndex,// input: sorted particle indices
    float4* oldPos,           // input: sorted position array
    float4* oldVel,           // input: sorted velocity array
    uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = oldPos[sortedIndex];
        float4 vel = oldVel[sortedIndex];

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }


}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
    float3 velA, float3 velB,
    float radiusA, float radiusB,
    float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring * (collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping * relVel;
        // tangential shear force
        force += params.shear * tanVel;
        // attraction
        force += attraction * relPos;
    }

    return force;
}


// collide two spheres using DEM method
__device__
float collideTriangle(float3 pos, float3 vel, float3 p0, float3 p1, float3 p2)
{
    float EPSILON = 0.0001f;
    float3 newPoint = pos + vel;
    float t;
 
    vel = normalize(vel);
    // compute plane's normal
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    // no need to normalize
    float3 N = cross(v0v1, v0v2);  //N 
    float theta = dot(N, pos-p0);

   // if (theta < 0)
     //   N = -N;
    float area2 = length(N);

    // Step 1: finding P
    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N, vel);
    //printf("enfaite \n");

    if (fabs(NdotRayDirection) < EPSILON)  //almost 0 (
        return 0.f;  //they are parallel so they don't intersect ! 

    // compute d parameter using equation 2
    float d = -dot(N, p0);

    // compute t (equation 3)
    t = -(dot(N, pos) + d) / NdotRayDirection;

    // check if the triangle is in behind the ray
    if (t < 0.f)return 0.f;  //the triangle is behind 

    // compute the intersection point using equation 1
    float3 P = pos + t * vel;

    // Step 2: inside-outside test
    float3 C;  //vector perpendicular to triangle's plane 

    // edge 0
    float3 edge0 = p1 - p0;
    float3 vp0 = P - p0;
    C = cross(edge0, vp0);
    if (dot(N, C) < 0.f) return 0.f;  //P is on the right side 

    // edge 1
   float3 edge1 = p2 - p1;
    float3 vp1 = P - p1;

    C = cross(edge1, vp1);

    if (dot(N, C) < 0.f)   return 0.f;  //P is on the right side 

    // edge 2
    float3 edge2 = p0 - p2;
    float3 vp2 = P - p2;
    C = cross(edge2, vp2);

    if (dot(N, C) < 0.f)  return 0.f;  //P is on the right side; 

    if (t * length(vel) < params.particleRadius )
        return t;
    else {
        return 0;

    }

}
__device__
float collideTriangle2(float3 pos, float3 vel, float3 p0, float3 p1, float3 p2)
{
    float EPSILON = 0.0001f;

    const float3 o = pos;
    const float3 d = vel;
    const float3 v0 = p0;
    const float3 v1 = p1;
    const float3 v2 = p2;

    const float3	  edge1 = v1 - v0;
    const float3	  edge2 = v2 - v0;

    const float3 pvec = cross(d, edge2);
    const float det = dot(edge1, pvec);

    if (det > -EPSILON && det < EPSILON) return 0.f;

    const float inv_det = 1.f / det;
    const float3 tvec = o - v0;
    const float3 qvec = cross(tvec, edge1);

    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.f || u > 1.f) return 0.f;

    float v = dot(d, qvec) * inv_det;
    if (v < 0.f || u + v > 1.f) return 0.f;

    float t = inv_det *dot(edge2, qvec);
    if (t > EPSILON )
    {
        return t;
    }

    else
        return 0.f;
}
// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
    uint    index,
    float3  pos,
    float3  vel,
    float4* oldPos,
    float4* oldVel,
    uint* cellStart,
    uint* cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
            }
        }
    }

    return force;
}

__global__
void getFlowForce(float4* newVel,               // output: new velocity
    float4* oldPos,               // input: sorted positions
    float4* oldVel,               // input: sorted velocities
    uint* gridParticleIndex,    // input: sorted particle indices
    uint* cellStart,
    uint* cellEnd,
    uint    numParticles,
    float inclinaison,
    float pumpForce) {

    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    uint originalIndex = gridParticleIndex[index];
    float3 pos = make_float3(oldPos[index]);
    float3 pousse = make_float3(0.f, 0.f, 0.0f);
    //printf("%f %f %f \n", pumpForce, lFactorForce, rFactorForce);
    if (pos.y < 0.f) {
        float Factor = 0.f;
        pousse = make_float3(0.f, pow(1.f - (1 + pos.y), 3) * 0.05f * pumpForce, 0.f);

    }
    newVel[originalIndex] += make_float4(pousse, 0.0f);
}
__global__
void collideD(float4* newVel,               // output: new velocity
    float4* oldPos,               // input: sorted positions
    float4* oldVel,               // input: sorted velocities
    uint* gridParticleIndex,    // input: sorted particle indices
    uint* cellStart,
    uint* cellEnd,
    uint    numParticles,
    float inclinaison
)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
            }
        }
    }

    // write new velocity back to original unsorted location
    uint originalIndex = gridParticleIndex[index];
    uint gridHash = calcGridHash(gridPos);

    float3 incForce = make_float3(0.f, 0.f, 0.0f);
    
    incForce += make_float3(inclinaison, 0.f, 0.f);

    
    newVel[originalIndex] = make_float4(vel+force, 0.0f);
}

#endif

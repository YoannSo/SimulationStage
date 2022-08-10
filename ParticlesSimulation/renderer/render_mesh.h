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

#include "../mesh/triangle_mesh_model.h"
#include <glm/glm/glm.hpp>

#ifndef __RENDER_MESH__
#define __RENDER_MESH__

class MeshRenderer
{
public:
    MeshRenderer(std::string p_name,std::string p_dirPath);
    ~MeshRenderer();

    void setPositions(float* pos, int numParticles);
    void setVertexBuffer(unsigned int vbo, int numParticles);
    void setColorBuffer(unsigned int vbo)
    {
        m_colorVBO = vbo;
    }

    enum DisplayMode
    {
        PARTICLE_POINTS,
        PARTICLE_SPHERES,
        PARTICLE_NUM_MODES,
        PARTICLE_LINE
    };

    void display(DisplayMode mode = PARTICLE_POINTS);
    void displayGrid();

    void setPointSize(float size)
    {
        m_pointSize = size;
    }
    void setParticleRadius(float r)
    {
        m_particleRadius = r;
    }
    void setColorMode(float mode) {
        m_color_mode = mode;
    }
    void setFOV(float fov)
    {
        m_fov = fov;
    }
    void setWindowSize(int w, int h)
    {
        m_window_w = w;
        m_window_h = h;
    }

    void takeScreenshot(int i);
    void loadMesh(const std::string& p_name, const std::string& p_path);
    void setMVMatrix(float* p_MV) {
        _model.setMVMatrix(p_MV);
    }
    glm::mat4 setRotationMatrix(glm::mat4 p_rotMatrix) {
        _model.setRotationMatrix(p_rotMatrix);
    }
    void setProjectionMatrix(glm::mat4 projectionMatrix) {
        _model.setProjectionMatrix(projectionMatrix);
    }
    void setTransformation(glm::mat4 p_Transformation) {
        _model.setTransformation(p_Transformation);
    }
    glm::mat4 getTransformation() {
        return _model._transformation;
    }
    void setViewMatrix(glm::mat4 projectionMatrix){}
    void setCamPos(glm::vec4 p_camPos) {
        _model.setCamPos(p_camPos);
    }
    unsigned int* getEbo() {
        return _model.getEbo().data();
        
    }
    int getEboSize() {
        std::vector<unsigned int> temp = _model.getEbo();
        temp.shrink_to_fit();
        return temp.size();
    }
    void changeRenderState() {
        if (m_render)
            m_render = false;
        else
            m_render = true;
    }
    TriangleMeshModel& getModel() { return _model; }
protected: // methods
    void _initGL(std::string p_name, std::string p_dirPath);
    void _drawPoints();
    void _drawLines();
    GLuint _compileProgram();

protected: // data

    TriangleMeshModel _model;
    float* m_pos;
    int m_numParticles;

    float m_pointSize;
    float m_particleRadius;
    float m_fov;
    float m_color_mode = 1;
    int m_window_w, m_window_h;
    bool m_render = true;
    GLuint m_program;
    GLuint m_vbo;
    GLuint m_colorVBO;
};

#endif //__ RENDER_PARTICLES__

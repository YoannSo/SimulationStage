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


#include <math.h>
#include <assert.h>
#include <stdio.h>

 // OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>


#include "render_mesh.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

MeshRenderer::MeshRenderer(std::string p_name, std::string p_dirPath)
    : m_pos(0),
    m_numParticles(0),
    m_pointSize(1.0f),
    m_particleRadius(0.125f * 0.5f),
    m_program(0),
    m_vbo(0),
    m_colorVBO(0)
{
    _initGL(p_name,p_dirPath);
}

MeshRenderer::~MeshRenderer()
{
    m_pos = 0;
}

void MeshRenderer::setPositions(float* pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void MeshRenderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void MeshRenderer::_drawPoints()
{

    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);


        if (m_colorVBO)
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }
        glDrawArrays(GL_TRIANGLES, 0, m_numParticles);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}
void MeshRenderer::_drawLines()
{

    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);


        if (m_colorVBO)
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }
        glDrawArrays(GL_POINTS, 0, m_numParticles);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}
void MeshRenderer::display(DisplayMode mode)
{
    switch (mode)
    {
    case PARTICLE_LINE:
        glColor3f(1, 1, 1);
        _drawLines();
        break;
    case PARTICLE_POINTS:
        glColor3f(1, 1, 1);
        glPointSize(m_pointSize);
        _drawPoints();
        break;

    default:
    case PARTICLE_SPHERES:
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(m_program);
        glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov * 0.5f * (float)M_PI / 180.0f));
        glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

        glUniform1f(glGetUniformLocation(m_program, "colorMode"), m_color_mode);

        glColor3f(1, 1, 1);
        _drawPoints();

        glUseProgram(0);
        break;
    }
}

GLuint
MeshRenderer::_compileProgram(const char* vsource, const char* fsource)
{

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}


void MeshRenderer::_initGL(std::string p_name,std::string p_dirPath)
{
    _model.load(p_name, p_dirPath);
    m_program = _compileProgram(testVertex, testPixel);

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}

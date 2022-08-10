#ifndef __TRIANGLE_MESH_MODEL_HPP__
#define __TRIANGLE_MESH_MODEL_HPP__

#include "triangle_mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <string>
#include <glm/glm/gtc/type_ptr.hpp>
#include <glm/glm/gtx/string_cast.hpp>
#include <iostream>
	class TriangleMeshModel 
	{
	public:
		TriangleMeshModel() = default;
		virtual ~TriangleMeshModel() = default;

		// Load a 3D model with Assimp.
		void load(const std::string& p_name, const std::string& p_filePath);

		void render(const GLuint p_glProgram);

		void cleanGL();
		void setMVMatrix(float* p_MV) {
			_MVMatrix  = glm::make_mat4(p_MV);
		}
		glm::mat4 setRotationMatrix(glm::mat4 p_rotMatrix) {
			_rotMatrix = p_rotMatrix;
		}
		void setTransformation(glm::mat4 p_transfo) {
			_transformation = p_transfo;
		}
		void setProjectionMatrix(glm::mat4 p_projection) {
			_projectionMatrix = p_projection;
		}
		void setViewMatrix(glm::mat4 p_viewMatrix) {
			_viewMatrix = p_viewMatrix;
		}
		void setCamPos(glm::vec4 p_camPos) {
			_camPos = p_camPos;
		}
		std::vector<unsigned int> getEbo() {
			return _meshes[0]._indices;
		}
		inline float3 getAABBMin() {
			return _min;
		}
		inline float3 getAABBMax() {
			return _max;
		}
		bool aabbContain(float3 p);
		void changeAABB(float3 p);
	private:
		void	 _loadMesh(const aiMesh* const p_mesh, const aiScene* const p_scene);

	public:
		std::vector<TriangleMesh> _meshes;		   
		// Some stats.
		float3 _max=make_float3(-9999999,-9999999,-9999999);
		float3 _min= make_float3(9999999, 9999999, 9999999);
		int _nbTriangles = 0;
		int _nbVertices = 0;
		std::string _name = "";
		glm::mat4 _MVMatrix;
		glm::mat4 _rotMatrix;

		glm::mat4 _projectionMatrix;
		glm::mat4 _viewMatrix;
		glm::vec4 _camPos;
		glm::mat4 _transformation= glm::mat4(1.f);
		std::string _dirPath = "";
	};

#endif // __TRIANGLE_MESH_MODEL_HPP__

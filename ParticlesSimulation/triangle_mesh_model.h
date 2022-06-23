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
		void setMVPMatrix(float* p_MVP) {
			_MVPMatrix  = glm::make_mat4(p_MVP);
		}
		void setProjectionMatrix(glm::mat4 p_projection) {
			_projectionMatrix = p_projection;
		}
		void setViewMatrix(glm::mat4 p_viewMatrix) {
			_viewMatrix = p_viewMatrix;
		}
		void setCamPos(glm::vec3 p_camPos) {
			_camPos = p_camPos;
		}
		std::vector<unsigned int> getEbo() {
			return _meshes[0]._indices;
		}
	private:
		void	 _loadMesh(const aiMesh* const p_mesh, const aiScene* const p_scene);

	public:
		std::vector<TriangleMesh> _meshes;		   // A model can contain several meshes.
		// Some stats.
		int _nbTriangles = 0;
		int _nbVertices = 0;
		std::string _name = "";
		glm::mat4 _MVPMatrix;
		glm::mat4 _projectionMatrix;
		glm::mat4 _viewMatrix;
		glm::vec3 _camPos;
		std::string _dirPath = "";
	};

#endif // __TRIANGLE_MESH_MODEL_HPP__

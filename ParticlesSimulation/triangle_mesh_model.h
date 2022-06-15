#ifndef __TRIANGLE_MESH_MODEL_HPP__
#define __TRIANGLE_MESH_MODEL_HPP__

#include "triangle_mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <string>

	class TriangleMeshModel 
	{
	public:
		TriangleMeshModel() = default;
		virtual ~TriangleMeshModel() = default;

		// Load a 3D model with Assimp.
		void load(const std::string& p_name, const std::string& p_filePath);

		void render(const GLuint p_glProgram);

		void cleanGL();
		
	private:
		void	 _loadMesh(const aiMesh* const p_mesh, const aiScene* const p_scene);

	public:
		std::vector<TriangleMesh> _meshes;		   // A model can contain several meshes.
		// Some stats.
		int _nbTriangles = 0;
		int _nbVertices = 0;
		std::string _name = "";

		std::string _dirPath = "";
	};

#endif // __TRIANGLE_MESH_MODEL_HPP__

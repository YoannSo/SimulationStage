#include "triangle_mesh_model.h"
#include <iostream>


	void TriangleMeshModel::load(const std::string& p_name, const std::string& p_filePath)
	{
		std::cout << "Loading: " << p_filePath << std::endl;
		Assimp::Importer importer;
		_name=p_name;
		// Read scene and triangulate meshes
		const aiScene* const scene
			= importer.ReadFile(p_filePath, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_GenUVCoords| aiProcess_JoinIdenticalVertices);

		// Importer options
		// Cf. http://assimp.sourceforge.net/lib_html/postprocess_8h.html.
		// Read scene :
		// - Triangulates meshes
		// - Computes vertex normals
		// - Computes tangent space (tangent and bitagent)

		if (scene == nullptr)
		{
			throw std::runtime_error("Fail to load file \" " + p_filePath + "\": " + importer.GetErrorString());
		}

		_meshes.reserve(scene->mNumMeshes);
		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			_loadMesh(scene->mMeshes[i], scene);
		}
		_meshes.shrink_to_fit();

		std::cout << "Done! "						//
			<< _meshes.size() << " meshes, "	//
			<< _nbTriangles << " triangles, " //
			<< _nbVertices << " vertices" << std::endl;
	}


	void TriangleMeshModel::render(const GLuint p_glProgram)
	{


		for (size_t i = 0; i < _meshes.size(); i++)
		{
			_meshes[i].render(p_glProgram,_MVMatrix,_projectionMatrix,_camPos,_rotMatrix,_transformation);
		}
	}

	void TriangleMeshModel::cleanGL()
	{
		for (size_t i = 0; i < _meshes.size(); i++)
		{
			_meshes[i].cleanGL();
		}
	}

	bool TriangleMeshModel::aabbContain(float3 p)
	{
		if (p.x <_max.x && p.y<_max.y && p.z<_max.z && p.x>_min.x && p.y>_min.y && p.z>_min.z)
			return true;
		return false;
	}

	void TriangleMeshModel::changeAABB(float3 p)
	{
	}

	void TriangleMeshModel::_loadMesh(const aiMesh* const p_mesh, const aiScene* const p_scene) {

		const std::string meshName = _name + "_" + std::string(p_mesh->mName.C_Str());

		std::cout << "-- Loading mesh: " << meshName << std::endl;

		// Load vertex attributes.
		std::vector<Vertex> vertices;
		vertices.resize(p_mesh->mNumVertices);
		for (unsigned int v = 0; v < p_mesh->mNumVertices; ++v)
		{
			Vertex& vertex = vertices[v];
			// Position.
			vertex._position.x = p_mesh->mVertices[v].x;
			vertex._position.y = p_mesh->mVertices[v].y;
			vertex._position.z = p_mesh->mVertices[v].z;
			if (!aabbContain(vertex._position)) {
				if (vertex._position.x > _max.x)
					_max.x = vertex._position.x;

				if (vertex._position.y > _max.y)
					_max.y = vertex._position.y;

				if (vertex._position.z > _max.z)
					_max.z = vertex._position.z;

				if (vertex._position.x < _min.x)
					_min.x = vertex._position.x;

				if (vertex._position.y < _min.y)
					_min.y = vertex._position.y;

				if (vertex._position.z < _min.z)
					_min.z = vertex._position.z;
	
			//	_max=glm::max(vertex._position, _max);
			}
			// Normal.
			vertex._normal.x = p_mesh->mNormals[v].x;
			vertex._normal.y = p_mesh->mNormals[v].y;
			vertex._normal.z = p_mesh->mNormals[v].z;
		}
<<<<<<< HEAD
=======
		printf("_max: %f %f %f _min %f %f %f \n", _max.x, _max.y, _max.z, _min.x, _min.y, _min.z);
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
			// Load indices.
			std::vector<unsigned int> indices;
			indices.resize(p_mesh->mNumFaces * 3); // Triangulated.
			for (unsigned int f = 0; f < p_mesh->mNumFaces; ++f)
			{
				const aiFace& face = p_mesh->mFaces[f];
				const unsigned int f3 = f * 3;
				indices[f3] = face.mIndices[0];
				indices[f3 + 1] = face.mIndices[1];
				indices[f3 + 2] = face.mIndices[2];
			}



			_nbTriangles += p_mesh->mNumFaces;
			_nbVertices += p_mesh->mNumVertices;

			_meshes.push_back(TriangleMesh(meshName, vertices, indices, Material()));

			std::cout << "-- Done! "						  //
				<< indices.size() / 3 << " triangles, " //
				<< vertices.size() << " vertices." << std::endl;

		
	}
		

	


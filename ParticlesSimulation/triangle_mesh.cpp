#include "triangle_mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <iostream>

	TriangleMesh::TriangleMesh(const std::string& p_name,
		const std::vector<Vertex>& p_vertices,
		const std::vector<unsigned int>& p_indices) :
		_name(p_name),
		_vertices(p_vertices), _indices(p_indices)
	{
		_vertices.shrink_to_fit();
		_indices.shrink_to_fit();
		_setupGL();
	}

	void TriangleMesh::render(const GLuint p_glProgram) const
	{
		glBindVertexArray(this->_vao);
		glDrawElements(GL_TRIANGLES, this->_indices.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

	void TriangleMesh::cleanGL()
	{	// on clean les vao,vbo etc
		glDisableVertexArrayAttribEXT(_vao, 0);
		glDisableVertexArrayAttribEXT(_vao, 1);
		glDeleteVertexArrays(1, &_vao);
		glDeleteBuffers(1, &_vbo);
		glDeleteBuffers(1, &_ebo);
	}

	// ici cela va etre differents car il faut calculer un offset
	void TriangleMesh::_setupGL()
	{
		// creation vbo
		glGenBuffers(1, &_vbo);
		glBufferData(_vbo, _vertices.size() * sizeof(Vertex), _vertices.data(), GL_STATIC_DRAW);
		//// creation ebo
		glGenBuffers(1, &_ebo);
		glBufferData(_ebo, _indices.size() * sizeof(unsigned int), _indices.data(), GL_STATIC_DRAW);

		//// creation vao
		glGenVertexArrays(1, &_vao);
		GLuint index_pos = 0;
		GLuint index_normal = 1;
		//// lie vao et vbo
		
		//// chaque id pour un atribut diffenrents
		//// 0: Cela va etre pour la position
		
		/*glEnableVertexArrayAttrib(_vao, 0);
		glVertex(_vao,
			0,
			3,// 3 pour le nombre de valeurs vec3

			GL_FLOAT, // GL_Float car on traite des flottant vec3F
			GL_FALSE,
			offsetof(Vertex, _position)); // offset a utilis*/
		//glVertexArrayAttribBinding(_vao, 0, 0);

		//// 1: Cela va etre pour la position
		//glEnableVertexArrayAttrib(_vao, 1);
		//glVertexArrayAttribFormat(_vao, 1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, _normal));
		//glVertexArrayAttribBinding(_vao, 1, 0);

		//// 2: Cela va etre pour la texCoor
		//glEnableVertexArrayAttrib(_vao, 2);
		//glVertexArrayAttribFormat(_vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, _texCoords));
		//glVertexArrayAttribBinding(_vao, 2, 0);

		//// 3: Cela va etre pour la tangent
		//glEnableVertexArrayAttrib(_vao, 3);
		//glVertexArrayAttribFormat(_vao, 3, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, _tangent));
		//glVertexArrayAttribBinding(_vao, 3, 0);

		//// 4: Cela va etre pour la bitangent
		//glEnableVertexArrayAttrib(_vao, 4);
		//glVertexArrayAttribFormat(_vao, 4, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, _bitangent));
		//glVertexArrayAttribBinding(_vao, 4, 0);


		//// on lie avec le vao
		//glVertexArrayElementBuffer(_vao, _ebo);
	}

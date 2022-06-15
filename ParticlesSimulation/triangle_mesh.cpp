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
		glBindVertexArray(_vao); /*bind VAO avec le programme*/
		//lancement du rendu dans les shaders
		glDrawElements(GL_TRIANGLES, _indices.size(), GL_UNSIGNED_INT, 0);

		glBindVertexArray(0); /*debind VAO*/
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
		glCreateBuffers(1, &_vbo);
		glNamedBufferData(_vbo, _vertices.size() * sizeof(Vertex), _vertices.data(), GL_STATIC_DRAW);
		// creation ebo
		glCreateBuffers(1, &_ebo);
		glNamedBufferData(_ebo, _indices.size() * sizeof(unsigned int), _indices.data(), GL_STATIC_DRAW);

		// creation vao
		glCreateVertexArrays(1, &_vao);
		GLuint index_pos = 0;
		GLuint index_normal = 1;
		glVertexArrayVertexBuffer(_vao, 0, _vbo, 0, sizeof(Vertex));

		// positions
		glEnableVertexArrayAttrib(_vao, index_pos);
		glVertexArrayAttribFormat(_vao,
			index_pos,
			3 /*car Vec(3)f*/,
			GL_FLOAT /*car Vec3(f)*/,
			GL_FALSE, /*non normalisé*/
			offsetof(Vertex, _position));
		glVertexArrayAttribBinding(_vao, index_pos, 0);

		// normale
		glEnableVertexArrayAttrib(_vao, index_normal);
		glVertexArrayAttribFormat(_vao,
			index_normal,
			3 /*car Vec(3)f*/,
			GL_FLOAT /*car Vec3(f)*/,
			GL_FALSE, /*non normalisé*/
			offsetof(Vertex, _normal));
		glVertexArrayAttribBinding(_vao, index_normal, 0);


		// liaison vao avec ebo
		glVertexArrayElementBuffer(_vao, _ebo);
	}

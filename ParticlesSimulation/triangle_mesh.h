#ifndef __TRIANGLE_MESH_HPP__
#define __TRIANGLE_MESH_HPP__
#include <vector>
#include <string>
#include <GL/glew.h>
#include <vector_types.h>

struct Vertex
{
	float3 _position;
	float3 _normal;

	//Vec2f _texCoords; coord des texture

	//Vec3f _tangent; calculer tbn, scale correctement, quand le scale n'est pas uniforme
	//Vec3f _bitangent;
};

class TriangleMesh
{
public:
	TriangleMesh() = delete;
	TriangleMesh(const std::string& p_name,
		const std::vector<Vertex>& p_vertices,
		const std::vector<unsigned int>& p_indices);

	~TriangleMesh() = default;

	void render(const GLuint p_glProgram) const;
	
	void cleanGL();

private:
	void _setupGL();

public:
	std::string _name = "Unknown";

	// ================ Geometric data.
	std::vector<Vertex>		  _vertices;
	std::vector<unsigned int> _indices;
	
	std::vector<float3>		  _position;
	std::vector<float3>		  _normal;



	// ================ GL data.
	GLuint _vao = GL_INVALID_INDEX; // Vertex Array Object
	GLuint _vbo = GL_INVALID_INDEX; // Vertex Buffer Object
	GLuint _ebo = GL_INVALID_INDEX; // Element Buffer Object
};
 // namespace M3D_ISICG
#endif // __TRIANGLE_MESH_HPP__
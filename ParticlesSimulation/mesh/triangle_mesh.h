#ifndef __TRIANGLE_MESH_HPP__
#define __TRIANGLE_MESH_HPP__
#include <vector>
#include <string>
#include <GL/glew.h>
#include <vector_types.h>
#include <helper_math.h>
#include <glm/glm/glm.hpp>
struct Vertex
{
	float3 _position;
	float3 _normal;
};
struct Material
{
	float _ambient[3] = { 1.0, 1.0, 1.0 };
	float _diffuse[3] = { 0.f, 0.f, 0.f };
	float _specular[3] = { 0.f, 0.f, 0.f };
	float _shininess = 0.f;

};
class TriangleMesh
{
public:
	TriangleMesh() = delete;
	TriangleMesh(const std::string& p_name,
		const std::vector<Vertex>& p_vertices,
		const std::vector<unsigned int>& p_indices, const Material& p_material);

	~TriangleMesh() = default;

	void render(const GLuint p_glProgram, glm::mat4 p_MVPMatrix, glm::mat4 p_ProjectionMatrix,glm::vec4 p_camPos,glm::mat4 p_rotMatrix,glm::mat4 p_transfo) const;
	
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
	Material _material;



	// ================ GL data.
	GLuint _vao = GL_INVALID_INDEX; // Vertex Array Object
	GLuint _vbo = GL_INVALID_INDEX; // Vertex Buffer Object
	GLuint _ebo = GL_INVALID_INDEX; // Element Buffer Object
};
 // namespace M3D_ISICG
#endif // __TRIANGLE_MESH_HPP__
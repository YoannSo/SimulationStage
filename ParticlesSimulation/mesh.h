#ifndef __RT_ISICG_MESH__
#define __RT_ISICG_MESH__
#include <vector>
#include <string>
#include <vector_types.h>
#include <helper_math.h>
namespace RT_ISICG
{
	class Mesh 
	{

	public:
		Mesh() = delete;
		Mesh(const std::string& p_name)  {}
		virtual ~Mesh() = default;
		const size_t getNbTriangles() const { return _triangles.size(); }
		const size_t getNbVertices() const { return _vertices.size(); }
		inline void	 addTriangle(const unsigned int p_v0, const unsigned int p_v1, const unsigned int p_v2)
		{
			_triangles.emplace_back(TriangleMeshGeometry(p_v0, p_v1, p_v2, this));
		};
		inline void addVertex(const float p_x, const float p_y, const float p_z)
		{			_vertices.emplace_back(p_x, p_y, p_z);
		}
		inline void addNormal(const float p_x, const float p_y, const float p_z)
		{
			_normals.emplace_back(p_x, p_y, p_z);
		}
		inline void addUV(const float p_u, const float p_v) { _uvs.emplace_back(p_u, p_v); }

	private:
		std::vector<float3>				  _vertices;
		std::vector<float3>				  _normals;
		std::vector<float2>				  _uvs;
		std::vector<float3[9]> _triangles;
		std::string _name;
	};
} // namespace RT_ISICG
#endif // __RT_ISICG_TRIANGLE_MESH__
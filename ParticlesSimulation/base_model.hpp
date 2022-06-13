#ifndef __BASE_OBJECT_HPP__
#define __BASE_OBJECT_HPP__


#include <string>


	class BaseModel
	{
	public:
		BaseModel() = default;
		BaseModel(const std::string& p_name) : _name(p_name) {}

		virtual ~BaseModel() = default;


		virtual void cleanGL() = 0;

	public:
		std::string _name = "Unknown";
		float		_transformation[4][4];
	};
 // namespace M3D_ISICG

#endif // __BASE_OBJECT_HPP__

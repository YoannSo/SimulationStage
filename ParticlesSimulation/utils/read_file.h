#ifndef __READ_FILE_HPP__
#define __READ_FILE_HPP__

#include <fstream>
#include <sstream>
#include <string>


	static inline std::string readFile(const std::string& p_filePath)
	{
		std::ifstream ifs(p_filePath, std::ifstream::in);
		if (!ifs.is_open())
			throw std::ios_base::failure("Cannot open file: " + p_filePath);

		std::stringstream s;
		s << ifs.rdbuf();
		ifs.close();
		return s.str();
	}


#endif //__READ_FILE_HPP__

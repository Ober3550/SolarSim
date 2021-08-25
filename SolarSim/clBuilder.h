#pragma once
#include <CL/cl.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>


namespace
{
	struct clProgram
	{
		cl::Context context;
		cl::Program program;
		cl::Buffer buffer;
		cl::Kernel kernel;
	};

	inline static std::string getFileData(const std::string filename)
	{
		std::string data;
		std::getline(std::ifstream(filename), data, '\0');
		if (data.size() == 0)
		{
			std::cout << "No kernel found" << std::endl;
		}
		std::cout << std::endl << "Kernel: " << std::endl << data << std::endl << std::endl;
		return data;
	}
}
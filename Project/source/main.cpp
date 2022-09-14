#pragma once
#include "VulkanRenderer.h"



int main()
{
	VulkanRenderer app;

	try
	{
		app.Run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	

	return 0;
}
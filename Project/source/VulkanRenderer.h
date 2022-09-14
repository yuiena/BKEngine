#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <set>
#include <glm/glm.hpp>
#include <array>


static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

struct ImageTransition
{
	VkImage             image;
	VkAccessFlags       currentAccess;
	VkAccessFlags       newAccess;
	VkImageLayout       currentLayout;
	VkImageLayout       newLayout;
	uint32_t            currentQueueFamily;
	uint32_t            newQueueFamily;
	VkImageAspectFlags  aspect;
};


struct SwapchainParameters
{
	VkSwapchainKHR				handle;
	VkFormat                    format;
	VkExtent2D                  size;
	std::vector<VkImage>        images;
	std::vector<VkImageView>	imageViews;
	std::vector<VkFramebuffer>	framebuffers;
	void clear()
	{
		images.clear();
		imageViews.clear();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};


struct QueueParameters
{
	uint32_t  familyIndex = -1;
	VkQueue   handle;

	bool IsValide() { return (familyIndex != -1) ? true : false; }
};

struct QueueFamilyIndices
{
	QueueParameters graphic;
	QueueParameters compute;
	QueueParameters present;

	bool isComplete()
	{
		return (graphic.familyIndex != -1) && (compute.familyIndex != -1) && (graphic.familyIndex != -1) ? true : false;
	}
};


struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	/**
	 * @brief vertex 데이터를 하나의 배열에 포장한다.
	 */
	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{
			0,								// binding
			sizeof(Vertex),					// stride
			VK_VERTEX_INPUT_RATE_VERTEX,	// inputRate
		};
		//inputRate 
		// VK_VERTEX_INPUT_RATE_VERTEX : vertex 뒤의 다음 데이터 항목으로 이동
		// VK_VERTEX_INPUT_RATE_INSTANCE : 각 instance 후 다음 데이터 항목으로 이동 (instance 사용시 사용!)
		return bindingDescription;
	}

	/**
	 * @brief 위의 binding description에서 만들어진 vertex 데이터 청크에서 vertex attribute 추출!
	 */
	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;						// vertex별 데이터의 binding index
		attributeDescriptions[0].location = 0;						// vertex shader에서 input의 location index
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;	// attribute data의 byte크기를 암시적으로 정의
		attributeDescriptions[0].offset = offsetof(Vertex, pos);	// vertex별 데이터 시작 이후 byte offset 지정.

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};




class VulkanRenderer
{
public:
	void Run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}


private:
	void initWindow();
	void initVulkan();
	void createInstance();
	void createSurface();
	void createCommandPool();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createSwapchain();
	void createRenderPass();
	void createGraphicsPipeline();
	void createFramebuffers();
	void createCommandBuffer();
	void createSyncObjects();
	/**
	 * @brief 
	 */
	void createVertexBuffer();

	void mainLoop();
	void cleanup();
	void drawFrame();

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
	void setImageMemoryBarrier(VkCommandBuffer command_buffer, VkPipelineStageFlags generating_stages, VkPipelineStageFlags consuming_stages, std::vector<ImageTransition> image_transitions);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	const QueueFamilyIndices& findQueueFamilies(VkPhysicalDevice device);
	bool isDeviceSuitable(VkPhysicalDevice device);
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
	VkShaderModule createShaderModule(const std::vector<char>& code);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	VkInstance _instance;

	VkDevice			_logicalDevice;
	VkPhysicalDevice	_physicalDevice = VK_NULL_HANDLE;

	VkCommandPool	_commandPool;
	VkCommandBuffer _commandBuffer;

	VkQueue _graphicQueue;
	VkQueue _computeQueue;
	VkQueue _presentQueue;

	VkSurfaceKHR _surface;

	SwapchainParameters _swapchain;
	VkRenderPass		_renderPass;

	VkPipelineLayout	_pipelineLayout;
	VkPipeline			_graphicsPipeline;

	VkPipelineLayout	_computePipelineLayout;
	VkPipeline			_computePipeline;

	VkSemaphore _imageAvailableSemaphore;
	VkSemaphore _renderFinishedSemaphore;
	VkFence		_inFlightFence;

	ImageTransition _imageTransition;

	VkBuffer _vertexBuffer;
	VkDeviceMemory _vertexBufferMemory;

	GLFWwindow* _window;

	const std::vector<Vertex> _vertices = {
	{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
	};
};
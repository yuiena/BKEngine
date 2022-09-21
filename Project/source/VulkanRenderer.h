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

	bool framebufferResized = false;

private:
	/**
	 * @brief window 초기화
	 */
	void initWindow();

	/**
	 * @brief vulkan 초기화
	 */
	void initVulkan();

	/**
	 * @brief
	 */
	void createInstance();

	/**
	 * @brief 렌더링된 이미지를 표시할 곳(surface)를 생성한다.
	 */
	void createSurface();

	/**
	 * @brief
	 */
	void createCommandPool();

	/**
	 * @brief Physical Device(GPU) 선택
	 */
	void pickPhysicalDevice();

	/**
	 * @brief hysical Device와 통신하기 위한 logical device 생성
	 */
	void createLogicalDevice();

	/**
	 * @brief swap chain 생성
	 * @details swap chain : screen에 출력되기 전에 기다리는 image queue
	 */
	void createSwapchain();

	/**
	 * @brief 어떤 컨텐츠로 렌더링 작업을 처리해야하는 지 등을 render pass에 랩핑하기 위해 생성.
	 */
	void createRenderPass();

	/**
	 * @brief graphic pipeline 생성
	 */
	void createGraphicsPipeline();

	/**
	 * @brief frame buffer 생성
	 */
	void createFramebuffers();

	/**
	 * @brief command buffer 생성
	 */
	void createCommandBuffer();

	/**
	 * @brief semaphore과 fense를 생성합니다.
	 * @details image를 획독했고 렌더링할 준비가 완료 됐다는 signal/ 렌더링 완료됐고 presentation이 발생할 수 있다는 signal을 위한 semaphore
	 */
	void createSyncObjects();

	/**
	 * @brief 
	 */
	void recreateSwapchain();

	/**
	 * @brief vertex buffer 생성
	 */
	void createVertexBuffer();

	/**
	 * @brief index buffer 생성
	 */
	void createIndexBuffer();

	/**
	 * @brief 버퍼 생성
	 * @param size			bufferSize
	 * @param usage			buffer의 데이터의 사용 용도 지정
	 * @param properties	VkMemoryPropertyFlags
	 * @param buffer		ouput buffer
	 * @param bufferMemory	ouput bufferMemory
	 */
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	/**
	 * @brief src에서 dst로 buffer를 복사합니다.
	 * @param srcBuffer
	 * @param dstBuffer
	 * @param size
	 */
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	/**
	 * @brief 
	 */
	void createUniformBuffers();

	/**
	 * @brief pipeline 생성 전에 shader안에서 사용되는 모든 descriptor binding에 대한 세부사항을 제공.
	 */
	void createDescriptorSetLayout();

	/**
	 * @brief
	 */
	void updateUniformBuffers(uint32_t currentImage);

	void createImageViews();

	void mainLoop();
	void cleanup();
	void cleanupSwapchain();
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
	std::vector<VkCommandBuffer> _commandBuffers;

	VkQueue _graphicQueue;
	VkQueue _computeQueue;
	VkQueue _presentQueue;

	VkSurfaceKHR _surface;

	SwapchainParameters _swapchain;
	VkRenderPass		_renderPass;

	VkDescriptorSetLayout _descriptorSetLayout;
	VkPipelineLayout	_pipelineLayout;
	VkPipeline			_graphicsPipeline;

	VkPipelineLayout	_computePipelineLayout;
	VkPipeline			_computePipeline;

	std::vector<VkSemaphore> _imageAvailableSemaphores;
	std::vector<VkSemaphore> _renderFinishedSemaphores;
	std::vector<VkFence>	_inFlightFences;
	uint32_t _currentFrame = 0;

	

	ImageTransition _imageTransition;

	//--------------------------------Shader
	VkBuffer _vertexBuffer;
	VkDeviceMemory _vertexBufferMemory;

	VkBuffer _indexBuffer;
	VkDeviceMemory _indexBufferMemory;

	std::vector<VkBuffer> _uniformBuffers;
	std::vector<VkDeviceMemory> _uniformBuffersMemory;

	GLFWwindow* _window;

	const std::vector<Vertex> _vertices = {
		{ { -0.5f, -0.5f },{ 1.0f, 0.0f, 0.0f } }, // red
		{ { 0.5f, -0.5f },{ 0.0f, 1.0f, 0.0f } }, // green
		{ { 0.5f, 0.5f },{ 0.0f, 0.0f, 1.0f } }, // blue
		{ { -0.5f, 0.5f },{ 1.0f, 1.0f, 1.0f } } // white
	};
	const std::vector<uint16_t> _indices = { 0, 1, 2, 2, 3, 0 };
};
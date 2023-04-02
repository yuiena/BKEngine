#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include "VulkanHelper.h"

#include <chrono>

#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <unordered_map>


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
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
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
	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;						// vertex별 데이터의 binding index
		attributeDescriptions[0].location = 0;						// vertex shader에서 input의 location index
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;	// attribute data의 byte크기를 암시적으로 정의
		attributeDescriptions[0].offset = offsetof(Vertex, pos);	// vertex별 데이터 시작 이후 byte offset 지정.

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const 
	{
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

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

	void setupDebugMessenger();

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
	 * @brief swapchain의 imageView를 생성합니다.
	 */
	void createSwapChainImageViews();

	/**
	 * @brief 어떤 컨텐츠로 렌더링 작업을 처리해야하는 지 등을 render pass에 랩핑하기 위해 생성.
	 */
	void createRenderPass();

	/**
	 * @brief pipeline 생성 전에 shader안에서 사용되는 모든 descriptor binding에 대한 세부사항을 제공.
	 */
	void createGraphicsDescriptorSetLayout();

	/**
	 * @brief graphic pipeline 생성
	 */
	void createGraphicsPipeline();


	/**
	 * @brief graphic pipeline 생성
	 */
	void createComputePipeline();


	/**
	 * @brief frame buffer 생성
	 */
	void createFramebuffers();

	/**
	 * @brief
	 */
	void createCommandPool(uint32_t familyIndex, VkCommandPool& commandPool);

	void createDepthResources();

	/**
	 * @brief Texture Image를 생성합니다.
	 */
	void createTextureImage(const std::string& path, VkImage& targetImage, VkDeviceMemory targetTextureMemory);

	/**
	 * @brief Texture의 ImageView를 생성합니다.
	 */
	void createTextureImageView();

	/**
	 * @brief TextureSampler를 생성합니다.
	 * @details sampler : shader가 image를 읽기 위한 형식(필터링 및 변환을 적용 가능)
	 */
	VkSampler createTextureSampler(VkSamplerAddressMode u, VkSamplerAddressMode v, VkSamplerAddressMode w, VkCompareOp compareOp, VkBorderColor borderColor);

	/**
	 * @brief
	 */
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);

	/**
	 * @brief
	 */
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
		VkImage& image, VkDeviceMemory& imageMemory);
	//, VkSharingMode sharingMode, uint32_t queueIndexCount, uint32_t* queueIndices);

	/**
	 * @brief Image를 올바른 layout으로 전환한다.
	 */
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);

	/**
	 * @brief
	 */
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	/**
	 * @brief vertex buffer 생성
	 */
	void createVertexBuffer();

	/**
	 * @brief index buffer 생성
	 */
	void createIndexBuffer();

	/**
	 * @brief
	 */
	void createUniformBuffers();

	/**
	 * @brief
	 */
	void createDescriptorPool();

	/**
	 * @brief
	 */
	void createDescriptorSets();

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
	 * @brief uniformBuffer를 업데이트 합니다.
	 */
	void updateUniformBuffer(uint32_t currentImage);

	/**
	 * @brief command buffer를 다시 기록하고 실행하는 도우미 함수 begin
	 */
	VkCommandBuffer beginSingleTimeCommands();

	/**
	 * @brief command buffer를 다시 기록하고 실행하는 도우미 함수 end
	 */
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);

	/**
	 * @brief 
	 */
	void dispatchCompute(VkCommandBuffer command_buffer, uint32_t x_size, uint32_t y_size, uint32_t z_size);

	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

	void loadModel();

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

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat findDepthFormat();

	VkInstance _instance;

	VkDevice			_logicalDevice;
	VkPhysicalDevice	_physicalDevice = VK_NULL_HANDLE;

	VkSurfaceKHR _surface;

	
	VkQueue _presentQueue;

	SwapchainParameters _swapchain;
	VkRenderPass		_renderPass;


	struct Graphic {
		VkCommandPool					commandPool;
		std::vector<VkCommandBuffer>	commandBuffers;
		VkDescriptorSetLayout			descriptorSetLayout;
		VkDescriptorPool				descriptorPool;
		std::vector<VkDescriptorSet>	descriptorSets;
		VkPipelineLayout				pipelineLayout;
		VkPipeline						pipeline;
		VkQueue							queue;
		VkSemaphore						semaphore;
	} _graphic;

	
	struct Compute {
		VkImageView						imageView;
		VkSampler						sampler;
		std::vector<VkCommandBuffer>	commandBuffers;	// Command buffer storing the dispatch commands and barriers
		VkCommandPool					commandPool;	// Use a separate command pool (queue family may differ from the one used for graphics)
		VkDescriptorSetLayout			descriptorSetLayout;
		std::vector<VkDescriptorSet>	descriptorSets;	// Compute shader bindings
		VkPipelineLayout				pipelineLayout;
		VkPipeline						pipeline;		// Compute pipelines for image filters
		VkQueue							queue;			// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkSemaphore						semaphore;		// Execution dependency between compute & graphic submission
	} _compute;

	std::vector<VkSemaphore> _presentCompeleteSemaphores;
	std::vector<VkSemaphore> _renderCompeletedSemaphores;
	std::vector<VkFence>	_inFlightFences;
	uint32_t _currentFrame = 0;

	// Pipeline cache object
	VkPipelineCache _pipelineCache;
	ImageTransition _imageTransition;

	//--------------------------------Shader
	VkBuffer _vertexBuffer;
	VkDeviceMemory _vertexBufferMemory;

	VkBuffer _indexBuffer;
	VkDeviceMemory _indexBufferMemory;

	std::vector<VkBuffer> _uniformBuffers;
	std::vector<VkDeviceMemory> _uniformBuffersMemory;
	std::vector<void*> _uniformBuffersMapped;

	//-------------------------------- Texture
	VkBuffer _stagingBuffer;
	VkDeviceMemory _stagingBufferMemory;

	uint32_t _mipLevels;
	VkImage _textureImage;
	VkDeviceMemory _textureImageMemory;

	VkImageView _textureImageView;
	VkSampler _textureSampler;

	VkImage _depthImage;
	VkDeviceMemory _depthImageMemory;
	VkImageView _depthImageView;
	//--------------------


	VkDebugUtilsMessengerEXT _debugMessenger;



	GLFWwindow* _window;

	std::vector<Vertex> _vertices;
		/*= {
		{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
		{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
		{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
	};*/
	std::vector<uint32_t> _indices;// = {0, 1, 2, 2, 3, 0};

	struct UniformBufferObject 
	{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};
};
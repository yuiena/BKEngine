#include "VulkanRenderer.h"
#include "FileSystem.h"


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::string engineName = "BKEngine";

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

const int MAX_FRAMES_IN_FLIGHT = 2;

#pragma region HelpFunction

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			return availableFormat;
	}
	return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
	for (const auto& availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			return availablePresentMode;
	}
	return VK_PRESENT_MODE_FIFO_KHR;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

bool checkAvailableInstanceExtensions(std::vector<VkExtensionProperties> & available_extensions)
{
	uint32_t extensions_count = 0;
	VkResult result = VK_SUCCESS;

	result = vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, nullptr);
	if ((result != VK_SUCCESS) || (extensions_count == 0))
	{
		std::cout << "Could not get the number of instance extensions." << std::endl;
		return false;
	}

	available_extensions.resize(extensions_count);
	result = vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, available_extensions.data());
	if ((result != VK_SUCCESS) || (extensions_count == 0))
	{
		std::cout << "Could not enumerate instance extensions." << std::endl;
		return false;
	}

	return true;
}

bool checkValidationLayerSupport()
{
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());



	bool layerFound = false;
	for (const char* layerName : validationLayers)
	{
		layerFound = false;
		for (const auto& layerProperties : availableLayers)
		{
			if (strcmp(layerName, layerProperties.layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound) return false;
	}

	return true;
}

std::vector<const char*> getRequiredExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

#pragma endregion



void VulkanRenderer::initVulkan()
{
	createInstance();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();

	createSwapchain();
	createImageViews();
	createRenderPass();

	//createDescriptorSetLayout();
	createGraphicsPipeline();
	createFramebuffers();
	createCommandPool();

	createVertexBuffer();
	createIndexBuffer();
	createCommandBuffer();
	createSyncObjects();
}

void VulkanRenderer::recreateSwapchain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(_window, &width, &height);
	while (width == 0 || height == 0) 
	{
		glfwGetFramebufferSize(_window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(_logicalDevice);

	cleanupSwapchain();

	createSwapchain();
	createImageViews();
	createFramebuffers();
}

void VulkanRenderer::cleanupSwapchain()
{
	for (auto framebuffer : _swapchain.framebuffers) {
		vkDestroyFramebuffer(_logicalDevice, framebuffer, nullptr);
	}

	for (auto imageView : _swapchain.imageViews)
	{
		vkDestroyImageView(_logicalDevice, imageView, nullptr);
	}

	vkDestroySwapchainKHR(_logicalDevice, _swapchain.handle, nullptr);
}

void VulkanRenderer::createUniformBuffers()
{
	//VkDeviceSize bufferSize = sizeof(UniformBufferObject);

	_uniformBuffers.resize(_swapchain.images.size());
	_uniformBuffersMemory.resize(_swapchain.images.size());

	for (size_t i = 0; i < _swapchain.images.size(); i++)
	{
		//createBuffer(_bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
	}
}

void VulkanRenderer::updateUniformBuffers(uint32_t currentImage)
{

}

void VulkanRenderer::createDescriptorSetLayout()
{
	// descriptor은 shader가 자유롭게 buffer나 image 같은 리소스를 액세스하는 것에 대한 방법.
	// 1. pipeline 생성동안 descriptor layout을 설정합니다.
	// 2. descriptor pool에서 descriptor set을 할당합니다.
	// 3. rendering동안 descriptor set을 바인드합니다.

	// binding 기술
	VkDescriptorSetLayoutBinding uboLayoutBinding = {
		0,									// shader binding
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,	// shader descriptorType
		1,									// descriptor Count 배열안의 몇 개의 변수가 있는지
		VK_SHADER_STAGE_VERTEX_BIT,			// Shader stage Flag
		nullptr,							// image sampler
	};
	
	VkDescriptorSetLayoutCreateInfo layoutInfo{
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,	// type
		nullptr,												// next
		0,														// DescriptorSetLayoutCreate Flags
		1,														// binding count
		&uboLayoutBinding										// bindings
	};

	// 모든 descriptor binding은 _descriptorSetLayout( VkDescriptorSetLayout )으로 결합된다.
	if (vkCreateDescriptorSetLayout(_logicalDevice, &layoutInfo, nullptr, &_descriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	// 어떤 descriptor가 shader에 사용될지 알려주기 위해 pipeline 생성 동안 descriptor set layout 지정
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,	// type
		nullptr,										// pNext
		0,												// VkPipelineLayoutCreateFlags
		1,												// setLayout Count
		&_descriptorSetLayout,							// pSetLayouts
		0,												// pushConstantRangeCount
		nullptr,										// VkPushConstantRange
	};

}

void VulkanRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	// command buffer 기록 시작.
	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to begin recording command buffer!");
	}


	//------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	ImageTransition imageTransitionForComputeShader = {
		_imageTransition.image,					  // VkImage              Image
		0,                                        // VkAccessFlags        CurrentAccess
		VK_ACCESS_SHADER_WRITE_BIT,               // VkAccessFlags        NewAccess
		VK_IMAGE_LAYOUT_UNDEFINED,                // VkImageLayout        CurrentLayout
		VK_IMAGE_LAYOUT_GENERAL,                  // VkImageLayout        NewLayout
		VK_QUEUE_FAMILY_IGNORED,                  // uint32_t             CurrentQueueFamily
		VK_QUEUE_FAMILY_IGNORED,                  // uint32_t             NewQueueFamily
		VK_IMAGE_ASPECT_COLOR_BIT                 // VkImageAspectFlags   Aspect
	};


	setImageMemoryBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, { imageTransitionForComputeShader });

	std::vector<VkDescriptorSet>	descriptor_sets;
	std::vector<uint32_t>			dynamic_offsets;

	// Dispatching compute work
	vkCmdBindDescriptorSets(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0,
		static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data(),
		static_cast<uint32_t>(dynamic_offsets.size()), dynamic_offsets.data());

	vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _computePipeline);

	vkCmdDispatch(_commandBuffer, _swapchain.size.width / 32 + 1, _swapchain.size.height / 32 + 1, 1);

	std::vector<ImageTransition> imageTransitionsForTransfer = {
		{
		_imageTransition.image,                 // VkImage              Image
		VK_ACCESS_SHADER_WRITE_BIT,             // VkAccessFlags        CurrentAccess
		VK_ACCESS_TRANSFER_READ_BIT,            // VkAccessFlags        NewAccess
		VK_IMAGE_LAYOUT_GENERAL,                // VkImageLayout        CurrentLayout
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,   // VkImageLayout        NewLayout
		VK_QUEUE_FAMILY_IGNORED,                // uint32_t             CurrentQueueFamily
		VK_QUEUE_FAMILY_IGNORED,                // uint32_t             NewQueueFamily
		VK_IMAGE_ASPECT_COLOR_BIT               // VkImageAspectFlags   Aspect
		},
		{
		_swapchain.images[imageIndex],          // VkImage              Image
		0,                                      // VkAccessFlags        CurrentAccess
		VK_ACCESS_TRANSFER_WRITE_BIT,           // VkAccessFlags        NewAccess
		VK_IMAGE_LAYOUT_UNDEFINED,              // VkImageLayout        CurrentLayout
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,   // VkImageLayout        NewLayout
		VK_QUEUE_FAMILY_IGNORED,				// uint32_t             CurrentQueueFamily ???????
		VK_QUEUE_FAMILY_IGNORED,				// uint32_t             NewQueueFamily ??????????
		VK_IMAGE_ASPECT_COLOR_BIT               // VkImageAspectFlags   Aspect
		},
	};

	setImageMemoryBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, imageTransitionsForTransfer);

	VkImageCopy imageCopy = {
		{								// VkImageSubresourceLayers     srcSubresource
			VK_IMAGE_ASPECT_COLOR_BIT,    // VkImageAspectFlags           aspectMask
			0,                            // uint32_t                     mipLevel
			0,                            // uint32_t                     baseArrayLayer
			1                             // uint32_t                     layerCount
			},
			{                             // VkOffset3D                   srcOffset
			0,                            // int32_t                      x
			0,                            // int32_t                      y
			0                             // int32_t                      z
			},
			{                             // VkImageSubresourceLayers     dstSubresource
			VK_IMAGE_ASPECT_COLOR_BIT,    // VkImageAspectFlags           aspectMask
			0,                            // uint32_t                     mipLevel
			0,                            // uint32_t                     baseArrayLayer
			1                             // uint32_t                     layerCount
			},
			{                             // VkOffset3D                   dstOffset
			0,                            // int32_t                      x
			0,                            // int32_t                      y
			0                             // int32_t                      z
			},
			{                             // VkExtent3D                   extent
			_swapchain.size.width,         // uint32_t                     width
			_swapchain.size.height,        // uint32_t                     height
			1                             // uint32_t                     depth
		}
	};
	vkCmdCopyImage(_commandBuffer, _imageTransition.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _swapchain.images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);

	ImageTransition image_transition_before_present = {
		_swapchain.images[imageIndex],             // VkImage              Image
		VK_ACCESS_TRANSFER_WRITE_BIT,             // VkAccessFlags        CurrentAccess
		VK_ACCESS_MEMORY_READ_BIT,                // VkAccessFlags        NewAccess
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,     // VkImageLayout        CurrentLayout
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,          // VkImageLayout        NewLayout
		VK_QUEUE_FAMILY_IGNORED,               // uint32_t             CurrentQueueFamily ????
		VK_QUEUE_FAMILY_IGNORED,               // uint32_t             NewQueueFamily ????
		VK_IMAGE_ASPECT_COLOR_BIT                 // VkImageAspectFlags   Aspect
	};

	setImageMemoryBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, { image_transition_before_present });
	*/
	//------------------------------------------------------------------------------------------------------------------------------------------------

	//render pass 를 시작해서 drawing을 시작하겠다는 의미.
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	//render pass 자신
	renderPassInfo.renderPass = _renderPass;
	// 바인딩 할 attachment
	renderPassInfo.framebuffer = _swapchain.framebuffers[imageIndex];
	// 렌더링 영역 크기 지정
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = _swapchain.size;
	//clear color 값
	VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	// 이제 render pass가 시작 되었습니다. command들이 기록하는 모든 함수는 근들이 가지고 있는
	// vkCmd 접두사로 알아볼 수 있습니다.
	// 첫 파라미터 : 항상 command를 기록 할 command buffer
	// 두번째 파라미터 : render pass 세부 항목
	// 세번쨰 파라미터 : render pass내에서 drawing command가 어케 제공되는지 제어.
	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	//graphics pipeline 바인딩
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphicsPipeline);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)_swapchain.size.width;
	viewport.height = (float)_swapchain.size.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = _swapchain.size;
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	//------------- Vertex Buffer binding 
	VkBuffer vertexBuffers[] = { _vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0/*first binding*/, 1/*binding count*/, vertexBuffers, offsets);

	//------------- Index Buffer binding 
	vkCmdBindIndexBuffer(commandBuffer, _indexBuffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(_indices.size())/*index count*/, 1/*instance count*/, 0/*first index*/, 0/*vertex offset*/, 0/*first instance*/);

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to record command buffer!");
	}
}

void VulkanRenderer::setImageMemoryBarrier(VkCommandBuffer              commandBuffer,
	VkPipelineStageFlags         generatingStages,
	VkPipelineStageFlags         consumingStages,
	std::vector<ImageTransition> imageTransitions)
{
	std::vector<VkImageMemoryBarrier> imageMemoryBarriers;

	for (auto & image_transition : imageTransitions)
	{
		imageMemoryBarriers.push_back({
		  VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,   // VkStructureType            sType
		  nullptr,                                  // const void               * pNext
		  _imageTransition.currentAccess,           // VkAccessFlags              srcAccessMask
		  _imageTransition.newAccess,               // VkAccessFlags              dstAccessMask
		  _imageTransition.currentLayout,           // VkImageLayout              oldLayout
		  _imageTransition.newLayout,               // VkImageLayout              newLayout
		  _imageTransition.currentQueueFamily,      // uint32_t                   srcQueueFamilyIndex
		  _imageTransition.newQueueFamily,          // uint32_t                   dstQueueFamilyIndex
		  _imageTransition.image,                   // VkImage                    image
		  {                                         // VkImageSubresourceRange    subresourceRange
			image_transition.aspect,                  // VkImageAspectFlags         aspectMask
			0,                                        // uint32_t                   baseMipLevel
			VK_REMAINING_MIP_LEVELS,                  // uint32_t                   levelCount
			0,                                        // uint32_t                   baseArrayLayer
			VK_REMAINING_ARRAY_LAYERS                 // uint32_t                   layerCount
		  }
			});
	}

	if (imageMemoryBarriers.size() > 0)
	{
		vkCmdPipelineBarrier(commandBuffer, generatingStages, consumingStages, 0, 0, nullptr, 0, nullptr,
			static_cast<uint32_t>(imageMemoryBarriers.size()), imageMemoryBarriers.data());
	}
}

void VulkanRenderer::drawFrame()
{
	/*
		draw하기 위해 해야하는 것.

		1. swap chain으로 부터 image 획득
		2. frame buffer에서 해당 image를 attachment로 command buffer 실행
		3. presentation을 위해 swap chain에 image 반환.
	*/
	vkWaitForFences(_logicalDevice, 1, &_inFlightFences[_currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(_logicalDevice, _swapchain.handle, UINT64_MAX, _imageAvailableSemaphores[_currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR) 
	{
		recreateSwapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	vkResetFences(_logicalDevice, 1, &_inFlightFences[_currentFrame]);

	//1. swap chain으로 부터 image 획득
	vkAcquireNextImageKHR(_logicalDevice, _swapchain.handle, UINT64_MAX, _imageAvailableSemaphores[_currentFrame], VK_NULL_HANDLE, &imageIndex);

	vkResetCommandBuffer(_commandBuffers[_currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
	recordCommandBuffer(_commandBuffers[_currentFrame], imageIndex);

	updateUniformBuffers(imageIndex);

	// queue submit(제출) 및 동기화
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;


	//semaphore들이 실행이 시작되기 전에 기다려아 하는지, pipeline의 stage(들)을 기다려야하는지 지정.
	VkSemaphore waitSemaphores[] = { _imageAvailableSemaphores[_currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	// 2. frame buffer에서 해당 image를 attachment로 command buffer 실행
	// swap chain image를 color attachment로 바인딩하는 command buffer 제출.
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &_commandBuffers[_currentFrame];

	//실행이 완료 됐을때 signal보낼 semaphore.
	VkSemaphore signalSemaphores[] = { _renderFinishedSemaphores[_currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	//위에서 셋팅했던 것들로 graphics queue에 command buffer 제출 가능.
	if (vkQueueSubmit(_graphicQueue, 1, &submitInfo, _inFlightFences[_currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer!");
	}

	// 3. presentation을 위해 swap chain에 image 반환.
	// frame을 drawing하는 마지막 단계.
	// 결과를 swap chain에게 다시 제출하여 최종적으로 화면에 표시하는 것이다.
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	// presentatin이 발생하기 전까지 기다릴 semaphore 지정
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	//image를 표시할 swap chain들과 각 swap chain의 index
	VkSwapchainKHR swapChains[] = { _swapchain.handle };
	presentInfo.swapchainCount = 1; //항상 1
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;

	// swap chain에게 image를 표시하라는 요청 제출!!
	vkQueuePresentKHR(_presentQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
	{
		framebufferResized = false;
		recreateSwapchain();
	}
	else if (result != VK_SUCCESS) 
	{
		throw std::runtime_error("failed to present swap chain image!");
	}

	_currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRenderer::mainLoop()
{
	while (!glfwWindowShouldClose(_window))
	{
		glfwPollEvents();
		drawFrame();
	}

	vkDeviceWaitIdle(_logicalDevice);
}

void VulkanRenderer::cleanup()
{

	cleanupSwapchain();

	vkDestroyPipeline(_logicalDevice, _graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(_logicalDevice, _pipelineLayout, nullptr);
	vkDestroyRenderPass(_logicalDevice, _renderPass, nullptr);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(_logicalDevice, _renderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(_logicalDevice, _imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(_logicalDevice, _inFlightFences[i], nullptr);
	}

	vkDestroyCommandPool(_logicalDevice, _commandPool, nullptr);

	vkDestroyBuffer(_logicalDevice, _vertexBuffer, nullptr);
	vkFreeMemory(_logicalDevice, _vertexBufferMemory, nullptr);

	vkDestroyBuffer(_logicalDevice, _indexBuffer, nullptr);
	vkFreeMemory(_logicalDevice, _indexBufferMemory, nullptr);

	vkDestroySwapchainKHR(_logicalDevice, _swapchain.handle, nullptr);
	vkDestroyDevice(_logicalDevice, nullptr);

	if (enableValidationLayers)
	{
		//DestroyDebugUtilsMessengerEXT(_instance, debugMessenger, nullptr);
	}
	vkDestroyDescriptorSetLayout(_logicalDevice, _descriptorSetLayout, nullptr);
	vkDestroySurfaceKHR(_instance, _surface, nullptr);
	vkDestroyInstance(_instance, nullptr);

	glfwDestroyWindow(_window);

	glfwTerminate();
}

void VulkanRenderer::createSyncObjects()
{
	_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		if (vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_imageAvailableSemaphores[i]) != VK_SUCCESS ||
			vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_renderFinishedSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(_logicalDevice, &fenceInfo, nullptr, &_inFlightFences[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create synchronization objects for a frame!");
		}
	}

}

void VulkanRenderer::createImageViews()
{
	//--------------------ImageView
	// swap chain 개수에 맞게 imageViews도 리사이즈
	_swapchain.imageViews.resize(_swapchain.images.size());


	for (size_t i = 0; i < _swapchain.images.size(); i++)
	{
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = _swapchain.images[i];
		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = _swapchain.format;
		// color channel을 섞을 수 있도록 해줌. (단색 텍스처를 쓴다면 모든 channel을 red로 매핑할 수도 있음.)
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		// 이미지의 용도, 어떤 부분을 액세스 해야하는지 기술
		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;

		// 다 설정 했으니 image View create!
		if (vkCreateImageView(_logicalDevice, &createInfo, nullptr, &_swapchain.imageViews[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image views!");
		}
	}
}

void VulkanRenderer::createCommandBuffer()
{
	_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	// VK_COMMAND_BUFFER_LEVEL_PRIMARY : 실행을 위해 queue에 제출될 수 있지만 다른 command buffer에서 호출 x
	// VK_COMMAND_BUFFER_LEVEL_SECONDARY : 직접 실행 x, primary command buffer에서 호출 o
	VkCommandBufferAllocateInfo allocInfo = {
	  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,   // VkStructureType          sType
	  nullptr,                                          // const void             * pNext
	  _commandPool,                                     // VkCommandPool            commandPool
	  VK_COMMAND_BUFFER_LEVEL_PRIMARY,                  // VkCommandBufferLevel     level
	 (uint32_t)_commandBuffers.size()												// uint32_t                 commandBufferCount
	};

	if (vkAllocateCommandBuffers(_logicalDevice, &allocInfo, &_commandBuffers[_currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate command buffers!");
	}
}

void VulkanRenderer::createFramebuffers()
{
	// swap chain의 모든 image를 위한 frame buffer들을 저장할 멤버 설정
	_swapchain.framebuffers.resize(_swapchain.imageViews.size());

	// imageView 개수만큼 framebuffer 생성
	for (size_t i = 0; i < _swapchain.imageViews.size(); i++)
	{
		VkImageView attachments[] = { _swapchain.imageViews[i] };

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		//frame buffer가 호환되는 render pass 사용(동일한 개수와 타입의 attachment를 사용해야 한다는 의미)
		framebufferInfo.renderPass = _renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = _swapchain.size.width;
		framebufferInfo.height = _swapchain.size.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(_logicalDevice, &framebufferInfo, nullptr, &_swapchain.framebuffers[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create framebuffer!");
		}
	}
}

void VulkanRenderer::createIndexBuffer()
{
	VkDeviceSize bufferSize = sizeof(_indices[0]) * _indices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, _indices.data(), (size_t)bufferSize);
	vkUnmapMemory(_logicalDevice, stagingBufferMemory);

	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _indexBuffer, _indexBufferMemory);

	copyBuffer(stagingBuffer, _indexBuffer, bufferSize);

	vkDestroyBuffer(_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(_logicalDevice, stagingBufferMemory, nullptr);
}

uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	//사용 가능한 메모리 유형에 대한 정보 쿼리
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) 
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) 
			return i;
	}
	throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanRenderer::createVertexBuffer()
{
	VkDeviceSize bufferSize = sizeof(_vertices[0]) * _vertices.size();

	// stagingBuffer : vertex 배열에서 데이터를 업로드기 위해 CPU 엑세스 가능한 메모리.
	
	// staging buffer는 vertex data를 high performance GPU local memory에 copy해서 렌더링 성능을 높이기 위함입니다. 
	// 그렇지 않은 경우, (잠재적으로 최적화되지 않음) CPU mappable buffer에 갇힐 것입니다.
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	// VK_BUFFER_USAGE_TRANSFER_SRC_BIT : Buffer가 memory transfer operation에서 source로 사용될 수 있다.
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, _vertices.data(), (size_t)bufferSize);
	vkUnmapMemory(_logicalDevice, stagingBufferMemory);

	// VK_BUFFER_USAGE_TRANSFER_DST_BIT : Buffer가 memory transfer operation에서 destination으로 사용될 수 있다.
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		_vertexBuffer, _vertexBufferMemory);

	//copy staging Buffer to Vertex Buffer
	copyBuffer(stagingBuffer, _vertexBuffer, bufferSize);

	vkDestroyBuffer(_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(_logicalDevice, stagingBufferMemory, nullptr);
}

void VulkanRenderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBufferAllocateInfo allocInfo{
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,	// struct type
		nullptr,										// next
		_commandPool,									// command pool
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,				// buffer Level
		1												// command buffer count
	};

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(_logicalDevice, &allocInfo, &commandBuffer);

	//command buffer 기록 시작
	VkCommandBufferBeginInfo beginInfo
	{
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,	// type
		nullptr,										// next
		VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,	// usage flag
		nullptr											// VkCommandBufferInheritanceInfo
	};
	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	// copy command
	VkBufferCopy copyRegion{
		0,		// src offset
		0,		// dst offset
		size	// size
	};
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	// command 기록 종료
	vkEndCommandBuffer(commandBuffer);

	// command buffer 실행
	VkSubmitInfo submitInfo{
		VK_STRUCTURE_TYPE_SUBMIT_INFO, // type
		nullptr,		// next
		0,				// wait semaphore count
		nullptr,		// wait semaphore
		nullptr,		// wait dst stage mask
		1,				// command buffer count
		&commandBuffer, // command buffer
		0,				// signal semaphore count
		nullptr			// signal semaphore
	};

	vkQueueSubmit(_graphicQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(_graphicQueue);

	//사용한 command buffer 삭제
	vkFreeCommandBuffers(_logicalDevice, _commandPool, 1, &commandBuffer);
}

void VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
	VkBufferCreateInfo bufferInfo =
	{
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,	// VkStructureType
		nullptr,								// next
		0,										// flag
		size,									// buffer byte 크기
		usage,									// VkBufferUsageFlags : buffer의 데이터가 어떤 용도로 사용되는지
		VK_SHARING_MODE_EXCLUSIVE,				// VkSharingMode
		0,										// queueFamilyIndexCount
		nullptr,								// pQueueFamilyIndices
	};

	// buffer 생성!
	if (vkCreateBuffer(_logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create vertex buffer!");
	}

	// 생성 된 buffer에 메모리 할당을 위해 memory requirement 쿼리
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(_logicalDevice, buffer, &memRequirements);

	// 메모리 할당하기 위한 정보 셋팅
	VkMemoryAllocateInfo allocInfo
	{
		VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,						// VkStructureType
		nullptr,													// next
		memRequirements.size,										// allocationSize
		findMemoryType(memRequirements.memoryTypeBits, properties) // memoryTypeIndex
	};
	

	// 실제 메모리 할당
	if (vkAllocateMemory(_logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate vertex buffer memory!");
	}

	vkBindBufferMemory(_logicalDevice, buffer, bufferMemory, 0);
}

VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code)
{
	//  VkShaderModule 오브젝트로 shader 랩핑.
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(_logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		throw std::runtime_error("failed to create shader module!");

	return shaderModule;
}

void VulkanRenderer::createGraphicsPipeline()
{
	//----------------------------------------------------------------------------------------------------------
	auto vertShaderCode = readFile("F:\\yuiena\\Engine\\Project\\res\\shader\\vert2.spv");
	auto fragShaderCode = readFile("F:\\yuiena\\Engine\\Project\\res\\shader\\frag2.spv");
	//auto computeShaderCode = readFile("F:\\yuiena\\Engine\\Project\\res\\shader\\shader.comp.spv");

	// 파이프라인에 코드를 전달하기 위해 VkShaderModule 오브젝트로 랩핑 해야함.
	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
	//VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

	// shader를 실제 사용하기 위해 VkPipelineShaderStageCreateInfo를 통해 pipeline state로 연결 해줌.
	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	/*VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
	computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computeShaderStageInfo.module = computeShaderModule;
	computeShaderStageInfo.pName = "main";*/

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };// , computeShaderStageInfo};

	/* ☆Input Assetmbler	- raw vertex 데이터 수집
		   Vertex Shader		- 모든 vertex에게 실행, model->screen 공간으로
		   Tessellation			- mesh 퀄리티를 올리기 위해 gemetry 세부화.(벽돌, 계단 등을 덜 평평해보이도록 사용)
		   Geometry shader		- 모든 primitive에 대해 실행되고 이를 버리거나 들어온 것 보다 더 많은 primitive를 출력 가능.
								(tessellation과 유사하지만 더 유연, 보통 글카에서 성능이 안 좋아서 요새 잘 사용 x, metal엔 이 단계 없음.)
		   ☆Resterization		- primitive를 frament로 분해. fragment는 pixel 요소로 frame buffer를 채움.
								(화면 밖 frament 폐기, depth testing으로 인한 fragment도 폐기)
		   Fragment Shader		- 살아남은 fragment로부터 어떤 framebuffer에 쓰여질지 어떤 color, depth를 사용할지 결정.
		   ☆Color blending		- frame buffer안에 동일한 pixel로 매칭되는 여러 fragment를 혼홉한다.

		   ☆이 붙은 단계가 fixed-function : 파라메터를 통해 operation을 수정할 수 있게 해주지만 미리 정의 됨
		   나머지는 programmable.
	*/

	//vertex shader에 전달되는 vertex 데이터의 포맷을 기술.
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	const VkVertexInputBindingDescription& bindingDescription = Vertex::getBindingDescription();
	const auto& attributeDescriptions = Vertex::getAttributeDescriptions();

	// 데이터 간의 간격과 per-vertex인지 per-instance인지 여부 결정.
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	// vertex로부터 어떤 종류의 geometry를 그릴 것이냐 primitive restart가 활성화 되었는가?
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	// viewport : image에서 frame buffer로서의 위치?크기? transformation을 정의.
	// scissor : 실제로 screen에 pixel을 그리는 영역
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	// vertex shader의 geometry를 받아 fragment shader로 색칠할 fragment로 변환.
	// 여기서 depth testing, face culiing, scissor test 수행.
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	// true라면 near/far plane을 벗어난 fragment는 폐기되는 대신 clamp됨.
	// 이 설정은 shadow map같은 특별항 상황에서 유용. (GPU feature 활성화 필요)
	rasterizer.depthClampEnable = VK_FALSE;
	// ture시 geometry가 rasteraizer 단계 진행 x 
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	// fragment가 생성되는 방법 결정 
	// FILL : fragment로 채움 / LINE : 엣지를 선으로 그림 / POINT : vertex를 점으로 그림.
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	// fragment 선의 두께 
	rasterizer.lineWidth = 1.0f;
	// face culling 우형
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	// multi sampling 구성. anti-aliasing을 수행하기 위한 방법 중 하나.		
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	// depth와 stencil buffer를 사용하기 위해 구성.
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;

	// color blending 
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	// pipeline 재생성 없이 변경할 수 있는 것들( viewport, scissor, line widht, blend constant등 )이 있는데 원한다면 그들을 채워넣어야함.
	std::vector<VkDynamicState> dynamicStates =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();


	//shader에서 사용되는 uniform값은 global값으로 dynamic state 와 유사하게 shader 재생성 없이 drawing 시점에서 바꿀 수 있다.
	// 이 uniform은 VkPipelineLayout 오브젝트 생성을 통해 pipeline을 생성하는 동안 지정된다.
	// Descriptor set with storage image
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // VkStructureType                  sType
		nullptr,                                        // const void                     * pNext
		0,                                              // VkPipelineLayoutCreateFlags      flags
		0,												// uint32_t                         setLayoutCount
		nullptr,										// const VkDescriptorSetLayout    * pSetLayouts
		0,												// uint32_t                         pushConstantRangeCount
		nullptr											// const VkPushConstantRange      * pPushConstantRanges
	};


	if (vkCreatePipelineLayout(_logicalDevice, &pipelineLayoutInfo, nullptr, &_pipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create pipeline layout!");
	}

	/*
	이전에 만든 모든 구조체와 오브젝트를 조합하여 드디어 graphics pipeline 생성 가능!
		- shader stages :shader module 생성
		- Fixed-function state : pipe line의 fixed-funtion을 정의
		- pieline layout : shader에 의해 참조되는 uniform과 push 변수는 draw time에 업뎃 가능
		- render pass :pipeline 단계에 참조하는 attachment와 그 사용법

	이 모든 것들이 조합되어  graphics pipeline의 기능을 완전히 정의합니다.
	*/
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = _pipelineLayout;
	pipelineInfo.renderPass = _renderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	// graphics pipeline create!
	if (vkCreateGraphicsPipelines(_logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_graphicsPipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create graphics pipeline!");
	}


	//----------------------------------------------------------------------------------------------------------Create Compute Pipeline
	// Descriptor set with storage image for compute
	//VkDescriptorSetLayoutBinding layoutBinding = {
	//	 0,                                // uint32_t             binding
	//	 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, // VkDescriptorType     descriptorType
	//	 1,                                // uint32_t             descriptorCount
	//	 VK_SHADER_STAGE_COMPUTE_BIT,      // VkShaderStageFlags   stageFlags
	//	 nullptr                           // const VkSampler    * pImmutableSamplers
	//};

	//VkPipelineLayoutCreateInfo computePipelineLayoutInfo = {
	//  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,          // VkStructureType                  sType
	//  nullptr,                                                // const void                     * pNext
	//  0,                                                      // VkPipelineLayoutCreateFlags      flags
	//  0,//static_cast<uint32_t>(descriptor_set_layouts.size()),   // uint32_t                         setLayoutCount
	//  nullptr,//descriptor_set_layouts.data(),                          // const VkDescriptorSetLayout    * pSetLayouts
	//  0,//static_cast<uint32_t>(push_constant_ranges.size()),     // uint32_t                         pushConstantRangeCount
	//  nullptr//push_constant_ranges.data()                             // const VkPushConstantRange      * pPushConstantRanges
	//};

	//if (VK_SUCCESS != vkCreatePipelineLayout(_logicalDevice, &computePipelineLayoutInfo, nullptr, &_computePipelineLayout))
	//{
	//	throw std::runtime_error("Could not create pipeline layout.");

	//}

	//VkComputePipelineCreateInfo computePipelineInfo = {
	//	VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // VkStructureType                    sType
	//	nullptr,                                        // const void                       * pNext
	//	0,												// VkPipelineCreateFlags              flags
	//	shaderStages[2],								// VkPipelineShaderStageCreateInfo    stage
	//	_computePipelineLayout,                         // VkPipelineLayout                   layout
	//	VK_NULL_HANDLE,									// VkPipeline                         basePipelineHandle
	//	-1                                              // int32_t                            basePipelineIndex
	//};

	//if (vkCreateComputePipelines(_logicalDevice, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &_computePipeline) != VK_SUCCESS)
	//{
	//	throw std::runtime_error("failed to create pipeline layout!");
	//}

	//-----------------------------------------------------------------------------------------------------------------------------------------------

	// pipeline 생성 후 지워야 함.
	vkDestroyShaderModule(_logicalDevice, fragShaderModule, nullptr);
	vkDestroyShaderModule(_logicalDevice, vertShaderModule, nullptr);
	//vkDestroyShaderModule(_logicalDevice, computeShaderModule, nullptr);
}

void VulkanRenderer::createRenderPass()
{
	//swap chain 이미지들 중 하나를 나타내는 color buffer attachment
	// color(색상) buffer(버퍼) attachment(첨부물)
	//https://www.notion.so/VkAttachmentDescription-Manual-Page-774a0dde223c41939b99b4b4f04349c9
	VkAttachmentDescription colorAttachment{};
	//color attachment의 format은 swap chain image들의 format과 동일.
	colorAttachment.format = _swapchain.format;
	// multisampling 관련 작업 셋팅( 아직 안 함으로 1)
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

	//attatchmnet의 데이터가 렌더링 전/후에 무엇을 할 것인지 결정
	// LOAD : 기존 attachment 유지 / CLEAR : 지우기 / DONT_CARE : 기존 컨텐츠 undefined
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// STORE : 렌더링된 컨텐츠 저장 후 읽을 수 있음 / DONT_CARE : 렌더링 후 undefined
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	//stencil 관련 설정 (stencil 버퍼로 fragment 폐기시 사용. 여기선 사용x)
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	//Vulkan의 texture와 frame buffer는 특성 pixel format인 VkImage 오브젝트로 표현 됨.

	// render pass를 시작하기 전 상태의 image layout 상태 지정
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;//이전 이미지가 어떤 layout이었던 상관없단 뜻.
	// render pass가 끝났을 때 자동적으로 전환될 layout을 지정.
	// 렌더링 후에 swap chain을 통해 image를 presentation 할 것이기 때문에 아래와 같이 설정.
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	//sub pass는 하나 이상의 attachment를 참조함.
	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// 단일 render pass는 여러개의 sub pass로 구성되는데 sub pass는 
	// 이전 pass의 frame buffer 내용에 의존하는 후속 렌더링 작업입니다. (ex) post-processing)

	// 지금은 삼각형 하나 띄울거니까 단일 sub pass 유지.
	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	// attachment 배열과 sub pass를 사용하여 VkRenderPassCreateInfo 구조체를 채우고 생성 가능!
	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	if (vkCreateRenderPass(_logicalDevice, &renderPassInfo, nullptr, &_renderPass) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create render pass!");
	}
}

void VulkanRenderer::createSwapchain()
{
	/*
		Swap Chain은 윈도우 surface와 호환되지 않을 수 있음으로 많은 세부 항목에 대한 질의가 필요합니다.
		1. Basic Surface capabilites(swap chain의 최대/최소 이미지 개수, 이미지 w/h 최대 최소 값)
		2. surface format(pixel format, color space)
		3. 사용 가능한 presentation 모드
	*/
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_physicalDevice);

	_swapchain.clear();
	if (!_swapchain.handle)
		vkDestroySwapchainKHR(_logicalDevice, _swapchain.handle, nullptr);


	// 1. surface format(color depth)
	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	// 2. presentation mode(이미지를 화면에 스와핑하기 위한 조건)
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	// 3. swap extent(swap chain의 이미지 해상도)
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);


	// swap chain이 사용할 이미지 개수도 정한다. + 1 은 일단 최소 1개 이상의 이미지를 요청하겠다는 의미이다.
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	//이제 Vulkan 오브젝트 관례대로 swap chain의 구조체를 채워보자.
	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = _surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	// 이미지를 구성하는 layer의 양 (3D APP을 개발하지 않는 이상 항상 1임.)
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphic.familyIndex, indices.present.familyIndex };

	// Graphic queue family와 Presentation queue family가 다른 경우 graphic queue에서 
	// swap chain 이미지, presentation queue에 그 이미지를 제출하게 됨.

	// Queue family간에 사용되는 S.C 이미지 핸들링 방법을 지정함.
	if (indices.graphic.familyIndex != indices.present.familyIndex)
	{
		// 명시적인 소유권 전송 없이 이미지는 여러 Queue Family에서 사용 가능.
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		// 이미지를 한 Queue family에서 소유하고 다른 Q.F에서 사용하려는 경우 명시적으로 소유권 전송.
		// 이 옵션은 최상의 성능을 제공 함.
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	// swap chain의 trasnform( ex: 90도 회전.. 수평 플립 등). 그대로 둘거면 current 하면 됨.
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	//윈도우 시스템에서 다른 윈도우와 블렌딩시 알파 채널 사용할 건가를 지정.
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;// 알파 채널 무시
	createInfo.presentMode = presentMode;
	// 가려진 픽셀을 신경쓰지 않겠다는 뜻
	createInfo.clipped = VK_TRUE;

	// 윈도우 리사이징할 때 등 이전 swap chain을 참조하기 위해 이 필드를 지정해야 함.
	// 복잡함으로 이것은 일단 null로 둠.(없을 시 항상 새로 생성)
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	// 모두 지정했으니 swap chain create!
	if (vkCreateSwapchainKHR(_logicalDevice, &createInfo, nullptr, &_swapchain.handle) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create swap chain!");
	}

	// 먼저 imageCount를 통해 이미지 개수를 질의한 뒤
	vkGetSwapchainImagesKHR(_logicalDevice, _swapchain.handle, &imageCount, nullptr);
	// 컨테이너 크기를 조정하고
	_swapchain.images.resize(imageCount);
	// 마지막으로 이를 다시 호출하여 핸들을 얻어온다. 이는 얼마든지 더 많은 수의 swapChain을 생성할 수 있기 떄문이다.
	vkGetSwapchainImagesKHR(_logicalDevice, _swapchain.handle, &imageCount, _swapchain.images.data());

	_swapchain.format = surfaceFormat.format;
	_swapchain.size = extent;

	
}

void VulkanRenderer::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphic.familyIndex, indices.present.familyIndex };// , indices.compute.familyIndex};

	float queuePriority = 1.0f; // 0.0 ~ 1.0 사이에서 사용 가능. 우선 순위 할당.
	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		//Queue Family에서 우리가 원하는 queue의 개수를 기술.
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};

	// VkDeviceQueueCreateInfo, VkPhysicalDeviceFeatures를 통해 VkDeviceCreateInfo 생성 가능.
	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else
	{
		createInfo.enabledLayerCount = 0;
	}

	if (vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_logicalDevice) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	// 각 queue family에서 queue 핸들을 찾아온다. 
	// ( logical device, queue family, queue index, queue handle 저장할 포인터)
	vkGetDeviceQueue(_logicalDevice, indices.graphic.familyIndex, 0, &_graphicQueue);
	vkGetDeviceQueue(_logicalDevice, indices.present.familyIndex, 0, &_presentQueue);
	//vkGetDeviceQueue(_logicalDevice, indices.compute.familyIndex, 0, &_computeQueue);
}

void VulkanRenderer::createSurface()
{
	// surface ? 렌더링된 이미지를 표시할 곳
	//각 플랫폼별 다른 구현을 통해 Vulkan surface 생성
	if (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}

SwapChainSupportDetails VulkanRenderer::querySwapChainSupport(VkPhysicalDevice physicalDevice)
{
	SwapChainSupportDetails details;

	//1. Basic Surface capabilites(swap chain의 최대 / 최소 이미지 개수, 이미지 w / h 최대 최소 값)
	// 지정된 Physical Device(GPU)와 윈도우 Surface를 사용하여 지원되는 capability를 결정함.
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, _surface, &details.capabilities);

	//2. surface format(pixel format, color space)
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, _surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, _surface, &formatCount, details.formats.data());
	}

	//3. 사용 가능한 presentation 모드
	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, _surface, &presentModeCount, nullptr);

	if (presentModeCount != 0)
	{
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, _surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

bool VulkanRenderer::isDeviceSuitable(VkPhysicalDevice physicalDevice)
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

	bool swapChainAdequate = false;
	if (extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

void VulkanRenderer::pickPhysicalDevice()
{
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);

	if (deviceCount == 0)
	{
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());

	for (const auto& device : devices)
	{
		if (isDeviceSuitable(device))
		{
			_physicalDevice = device;
			break;
		}
	}

	if (_physicalDevice == VK_NULL_HANDLE)
		throw std::runtime_error("failed to find a suitable GPU!");
}

void VulkanRenderer::createInstance()
{
	std::vector<VkExtensionProperties> availableExtensions;
	if (!checkAvailableInstanceExtensions(availableExtensions)) return;

	if (enableValidationLayers && !checkValidationLayerSupport())
	{
		throw std::runtime_error("validation layers requested, but not available!");
	}

	VkApplicationInfo appInfo = {
	  VK_STRUCTURE_TYPE_APPLICATION_INFO,               // VkStructureType           sType
	  nullptr,                                          // const void              * pNext
	  engineName.c_str(),								// const char              * pApplicationName
	  VK_MAKE_VERSION(1, 0, 0),                         // uint32_t                  applicationVersion
	  engineName.c_str(),								// const char              * pEngineName
	  VK_MAKE_VERSION(1, 0, 0),                         // uint32_t                  engineVersion
	  VK_MAKE_VERSION(1, 0, 0)                          // uint32_t                  apiVersion
	};

	auto extensions = getRequiredExtensions();
	VkInstanceCreateInfo createInfo = {
	  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,           // VkStructureType           sType
	  nullptr,                                          // const void              * pNext
	  0,                                                // VkInstanceCreateFlags     flags
	  &appInfo,											// const VkApplicationInfo * pApplicationInfo
	  0,                                                // uint32_t                  enabledLayerCount
	  nullptr,                                          // const char * const      * ppEnabledLayerNames
	  static_cast<uint32_t>(extensions.size()),			// uint32_t                  enabledExtensionCount
	  extensions.data()									// const char * const      * ppEnabledExtensionNames
	};

	// setting validation layer
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		// 이벤트 심각도 지정
		debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		// 콜백이 호출되도록 하는 이벤트 유형 지정
		debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		//콜백 함수
		debugCreateInfo.pfnUserCallback = debugCallback;

		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else
	{
		createInfo.enabledLayerCount = 0;
		createInfo.pNext = nullptr;
	}

	// finally! create vulkan Instance!
	if (vkCreateInstance(&createInfo, nullptr, &_instance) != VK_SUCCESS)
		throw std::runtime_error("failed to create instance!");
}

const QueueFamilyIndices& VulkanRenderer::findQueueFamilies(VkPhysicalDevice device)
{
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	// queue family 리스트를 얻어옴
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies)
	{
		//VK_QUEUE_GRAPHICS_BIT 를 지원하는 최소 하나의 queue family를 찾아야 함.
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			indices.graphic.familyIndex = i;

		//https://gist.github.com/sheredom/523f02bbad2ae397d7ed255f3f3b5a7f
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)
			indices.compute.familyIndex = i;

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &presentSupport);

		if (presentSupport)
			indices.present.familyIndex = i; 

		if (indices.isComplete())
			break;

		i++;
	}

	if (!indices.isComplete())
		throw std::runtime_error("failed to set QueueFamilyIndices!");

	return indices;
}

void VulkanRenderer::createCommandPool()
{
	if (_physicalDevice == NULL)
	{
		throw std::runtime_error("_physicalDevice is NULL!");
	}

	// command buffer는 queue중 하나에 제출함으로써 실행 됨. 고로 queue를 가져 옴.
	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

	// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 새로운 command가 매우 자주 기록.
	// VK_COMMNAD_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : command buffer가 개별적으로 재기록 될 수 있음.
	VkCommandPoolCreateInfo poolInfo = {
	  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,		// VkStructureType              sType
	  nullptr,											// const void                 * pNext
	  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,  // VkCommandPoolCreateFlags     flags
	  indices.graphic.familyIndex							// uint32_t                     queueFamilyIndex
	};

	VkResult result = vkCreateCommandPool(_logicalDevice, &poolInfo, nullptr, &_commandPool);
	if (VK_SUCCESS != result)
	{
		std::cout << "Could not create command pool." << std::endl;
		return;
	}
}

static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	auto app = reinterpret_cast<VulkanRenderer*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}

void  VulkanRenderer::initWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	_window = glfwCreateWindow(WIDTH, HEIGHT, engineName.c_str(), nullptr, nullptr);
	glfwSetWindowUserPointer(_window, this);
	glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);
}


VkExtent2D VulkanRenderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(_window, &width, &height);

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}
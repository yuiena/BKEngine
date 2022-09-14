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
	createRenderPass();
	createGraphicsPipeline();
	createFramebuffers();

	createCommandPool();
	createCommandBuffer();
	createSyncObjects();
}


void VulkanRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	// command buffer ��� ����.
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

	//render pass �� �����ؼ� drawing�� �����ϰڴٴ� �ǹ�.
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	//render pass �ڽ�
	renderPassInfo.renderPass = _renderPass;
	// ���ε� �� attachment
	renderPassInfo.framebuffer = _swapchain.framebuffers[imageIndex];
	// ������ ���� ũ�� ����
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = _swapchain.size;
	//clear color ��
	VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	// ���� render pass�� ���� �Ǿ����ϴ�. command���� ����ϴ� ��� �Լ��� �ٵ��� ������ �ִ�
	// vkCmd ���λ�� �˾ƺ� �� �ֽ��ϴ�.
	// ù �Ķ���� : �׻� command�� ��� �� command buffer
	// �ι�° �Ķ���� : render pass ���� �׸�
	// ������ �Ķ���� : render pass������ drawing command�� ���� �����Ǵ��� ����.
	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	//graphics pipeline ���ε�
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

	// ------------ Vertex Buffer binding 
	VkBuffer vertexBuffers[] = { _vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

	vkCmdDraw(commandBuffer, 3, 1, 0, 0);

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
		draw�ϱ� ���� �ؾ��ϴ� ��.

		1. swap chain���� ���� image ȹ��
		2. frame buffer���� �ش� image�� attachment�� command buffer ����
		3. presentation�� ���� swap chain�� image ��ȯ.
	*/
	vkWaitForFences(_logicalDevice, 1, &_inFlightFence, VK_TRUE, UINT64_MAX);
	vkResetFences(_logicalDevice, 1, &_inFlightFence);

	//1. swap chain���� ���� image ȹ��
	uint32_t imageIndex;
	vkAcquireNextImageKHR(_logicalDevice, _swapchain.handle, UINT64_MAX, _imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

	vkResetCommandBuffer(_commandBuffer, /*VkCommandBufferResetFlagBits*/ 0);
	recordCommandBuffer(_commandBuffer, imageIndex);


	// queue submit(����) �� ����ȭ
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;


	//semaphore���� ������ ���۵Ǳ� ���� ��ٷ��� �ϴ���, pipeline�� stage(��)�� ��ٷ����ϴ��� ����.
	VkSemaphore waitSemaphores[] = { _imageAvailableSemaphore };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	// 2. frame buffer���� �ش� image�� attachment�� command buffer ����
	// swap chain image�� color attachment�� ���ε��ϴ� command buffer ����.
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &_commandBuffer;

	//������ �Ϸ� ������ signal���� semaphore.
	VkSemaphore signalSemaphores[] = { _renderFinishedSemaphore };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	//������ �����ߴ� �͵�� graphics queue�� command buffer ���� ����.
	if (vkQueueSubmit(_graphicQueue, 1, &submitInfo, _inFlightFence) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer!");
	}

	// 3. presentation�� ���� swap chain�� image ��ȯ.
	// frame�� drawing�ϴ� ������ �ܰ�.
	// ����� swap chain���� �ٽ� �����Ͽ� ���������� ȭ�鿡 ǥ���ϴ� ���̴�.
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	// presentatin�� �߻��ϱ� ������ ��ٸ� semaphore ����
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	//image�� ǥ���� swap chain��� �� swap chain�� index
	VkSwapchainKHR swapChains[] = { _swapchain.handle };
	presentInfo.swapchainCount = 1; //�׻� 1
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;

	// swap chain���� image�� ǥ���϶�� ��û ����!!
	vkQueuePresentKHR(_presentQueue, &presentInfo);
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
	vkDestroySemaphore(_logicalDevice, _renderFinishedSemaphore, nullptr);
	vkDestroySemaphore(_logicalDevice, _imageAvailableSemaphore, nullptr);
	vkDestroyFence(_logicalDevice, _inFlightFence, nullptr);

	vkDestroyCommandPool(_logicalDevice, _commandPool, nullptr);

	vkDestroyBuffer(_logicalDevice, _vertexBuffer, nullptr);

	//frame buffer�� image view��� render pass ���Ŀ� ���� �Ǿ� ��.
	for (auto framebuffers : _swapchain.framebuffers)
	{
		vkDestroyFramebuffer(_logicalDevice, framebuffers, nullptr);
	}

	vkDestroyPipeline(_logicalDevice, _graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(_logicalDevice, _pipelineLayout, nullptr);
	vkDestroyRenderPass(_logicalDevice, _renderPass, nullptr);

	for (auto imageView : _swapchain.imageViews)
	{
		vkDestroyImageView(_logicalDevice, imageView, nullptr);
	}

	vkDestroySwapchainKHR(_logicalDevice, _swapchain.handle, nullptr);
	vkDestroyDevice(_logicalDevice, nullptr);

	if (enableValidationLayers)
	{
		//DestroyDebugUtilsMessengerEXT(_instance, debugMessenger, nullptr);
	}

	vkDestroySurfaceKHR(_instance, _surface, nullptr);
	vkDestroyInstance(_instance, nullptr);

	glfwDestroyWindow(_window);

	

	glfwTerminate();
}

void VulkanRenderer::createSyncObjects()
{
	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_imageAvailableSemaphore) != VK_SUCCESS ||
		vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_renderFinishedSemaphore) != VK_SUCCESS ||
		vkCreateFence(_logicalDevice, &fenceInfo, nullptr, &_inFlightFence) != VK_SUCCESS) {
		throw std::runtime_error("failed to create synchronization objects for a frame!");
	}

}

void VulkanRenderer::createCommandBuffer()
{
	// VK_COMMAND_BUFFER_LEVEL_PRIMARY : ������ ���� queue�� ����� �� ������ �ٸ� command buffer���� ȣ�� x
	// VK_COMMAND_BUFFER_LEVEL_SECONDARY : ���� ���� x, primary command buffer���� ȣ�� o
	VkCommandBufferAllocateInfo allocInfo = {
	  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,   // VkStructureType          sType
	  nullptr,                                          // const void             * pNext
	  _commandPool,                                     // VkCommandPool            commandPool
	  VK_COMMAND_BUFFER_LEVEL_PRIMARY,                  // VkCommandBufferLevel     level
	  1													// uint32_t                 commandBufferCount
	};

	if (vkAllocateCommandBuffers(_logicalDevice, &allocInfo, &_commandBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate command buffers!");
	}
}

void VulkanRenderer::createFramebuffers()
{
	// swap chain�� ��� image�� ���� frame buffer���� ������ ��� ����
	_swapchain.framebuffers.resize(_swapchain.imageViews.size());

	// imageView ������ŭ framebuffer ����
	for (size_t i = 0; i < _swapchain.imageViews.size(); i++)
	{
		VkImageView attachments[] = { _swapchain.imageViews[i] };

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		//frame buffer�� ȣȯ�Ǵ� render pass ���(������ ������ Ÿ���� attachment�� ����ؾ� �Ѵٴ� �ǹ�)
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

uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	//��� ������ �޸� ������ ���� ���� ����
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
	VkBufferCreateInfo bufferInfo = 
	{
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,	// VkStructureType
		nullptr,								// next
		0,										// flag
		sizeof(_vertices[0]) * _vertices.size(),// buffer byte ũ��
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,		// VkBufferUsageFlags : buffer�� �����Ͱ� � �뵵�� ���Ǵ���
		VK_SHARING_MODE_EXCLUSIVE,				// VkSharingMode
		0,										// queueFamilyIndexCount
		nullptr,								// pQueueFamilyIndices
	};
	
	// buffer ����!
	if (vkCreateBuffer(_logicalDevice, &bufferInfo, nullptr, &_vertexBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create vertex buffer!");
	}

	// ���� �� buffer�� �޸� �Ҵ��� ���� memory requirement ����
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(_logicalDevice, _vertexBuffer, &memRequirements);

	// �޸� �Ҵ��ϱ� ���� ���� ����
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	// �׷���ī��� �Ҵ��� ���� ���� �ٸ� �޸� ������ ������ �� �������� �ùٸ� ������ �޸𸮸� ã�´�.
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	// ���� �޸� �Ҵ�
	if (vkAllocateMemory(_logicalDevice, &allocInfo, nullptr, &_vertexBufferMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate vertex buffer memory!");
	}

	vkBindBufferMemory(_logicalDevice, _vertexBuffer, _vertexBufferMemory, 0);

	void* data;
	vkMapMemory(_logicalDevice, _vertexBufferMemory, 0, bufferInfo.size, 0, &data);
	memcpy(data, _vertices.data(), (size_t)bufferInfo.size);
	vkUnmapMemory(_logicalDevice, _vertexBufferMemory);
}

VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code)
{
	//  VkShaderModule ������Ʈ�� shader ����.
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

	// ���������ο� �ڵ带 �����ϱ� ���� VkShaderModule ������Ʈ�� ���� �ؾ���.
	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
	//VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

	// shader�� ���� ����ϱ� ���� VkPipelineShaderStageCreateInfo�� ���� pipeline state�� ���� ����.
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

	/* ��Input Assetmbler	- raw vertex ������ ����
		   Vertex Shader		- ��� vertex���� ����, model->screen ��������
		   Tessellation			- mesh ����Ƽ�� �ø��� ���� gemetry ����ȭ.(����, ��� ���� �� �����غ��̵��� ���)
		   Geometry shader		- ��� primitive�� ���� ����ǰ� �̸� �����ų� ���� �� ���� �� ���� primitive�� ��� ����.
								(tessellation�� ���������� �� ����, ���� ��ī���� ������ �� ���Ƽ� ��� �� ��� x, metal�� �� �ܰ� ����.)
		   ��Resterization		- primitive�� frament�� ����. fragment�� pixel ��ҷ� frame buffer�� ä��.
								(ȭ�� �� frament ���, depth testing���� ���� fragment�� ���)
		   Fragment Shader		- ��Ƴ��� fragment�κ��� � framebuffer�� �������� � color, depth�� ������� ����.
		   ��Color blending		- frame buffer�ȿ� ������ pixel�� ��Ī�Ǵ� ���� fragment�� ȥȩ�Ѵ�.

		   ���� ���� �ܰ谡 fixed-function : �Ķ���͸� ���� operation�� ������ �� �ְ� �������� �̸� ���� ��
		   �������� programmable.
	*/

	//vertex shader�� ���޵Ǵ� vertex �������� ������ ���.
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	const VkVertexInputBindingDescription& bindingDescription = Vertex::getBindingDescription();
	const auto& attributeDescriptions = Vertex::getAttributeDescriptions();

	// ������ ���� ���ݰ� per-vertex���� per-instance���� ���� ����.
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	// vertex�κ��� � ������ geometry�� �׸� ���̳� primitive restart�� Ȱ��ȭ �Ǿ��°�?
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	// viewport : image���� frame buffer�μ��� ��ġ?ũ��? transformation�� ����.
	// scissor : ������ screen�� pixel�� �׸��� ����
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	// vertex shader�� geometry�� �޾� fragment shader�� ��ĥ�� fragment�� ��ȯ.
	// ���⼭ depth testing, face culiing, scissor test ����.
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	// true��� near/far plane�� ��� fragment�� ���Ǵ� ��� clamp��.
	// �� ������ shadow map���� Ư���� ��Ȳ���� ����. (GPU feature Ȱ��ȭ �ʿ�)
	rasterizer.depthClampEnable = VK_FALSE;
	// ture�� geometry�� rasteraizer �ܰ� ���� x 
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	// fragment�� �����Ǵ� ��� ���� 
	// FILL : fragment�� ä�� / LINE : ������ ������ �׸� / POINT : vertex�� ������ �׸�.
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	// fragment ���� �β� 
	rasterizer.lineWidth = 1.0f;
	// face culling ����
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	// multi sampling ����. anti-aliasing�� �����ϱ� ���� ��� �� �ϳ�.		
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	// depth�� stencil buffer�� ����ϱ� ���� ����.
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

	// pipeline ����� ���� ������ �� �ִ� �͵�( viewport, scissor, line widht, blend constant�� )�� �ִµ� ���Ѵٸ� �׵��� ä���־����.
	std::vector<VkDynamicState> dynamicStates =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();


	//shader���� ���Ǵ� uniform���� global������ dynamic state �� �����ϰ� shader ����� ���� drawing �������� �ٲ� �� �ִ�.
	// �� uniform�� VkPipelineLayout ������Ʈ ������ ���� pipeline�� �����ϴ� ���� �����ȴ�.
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
	������ ���� ��� ����ü�� ������Ʈ�� �����Ͽ� ���� graphics pipeline ���� ����!
		- shader stages :shader module ����
		- Fixed-function state : pipe line�� fixed-funtion�� ����
		- pieline layout : shader�� ���� �����Ǵ� uniform�� push ������ draw time�� ���� ����
		- render pass :pipeline �ܰ迡 �����ϴ� attachment�� �� ����

	�� ��� �͵��� ���յǾ�  graphics pipeline�� ����� ������ �����մϴ�.
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

	VkPipelineLayoutCreateInfo computePipelineLayoutInfo = {
	  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,          // VkStructureType                  sType
	  nullptr,                                                // const void                     * pNext
	  0,                                                      // VkPipelineLayoutCreateFlags      flags
	  0,//static_cast<uint32_t>(descriptor_set_layouts.size()),   // uint32_t                         setLayoutCount
	  nullptr,//descriptor_set_layouts.data(),                          // const VkDescriptorSetLayout    * pSetLayouts
	  0,//static_cast<uint32_t>(push_constant_ranges.size()),     // uint32_t                         pushConstantRangeCount
	  nullptr//push_constant_ranges.data()                             // const VkPushConstantRange      * pPushConstantRanges
	};

	if (VK_SUCCESS != vkCreatePipelineLayout(_logicalDevice, &computePipelineLayoutInfo, nullptr, &_computePipelineLayout))
	{
		throw std::runtime_error("Could not create pipeline layout.");

	}

	VkComputePipelineCreateInfo computePipelineInfo = {
		VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // VkStructureType                    sType
		nullptr,                                        // const void                       * pNext
		0,												// VkPipelineCreateFlags              flags
		shaderStages[2],								// VkPipelineShaderStageCreateInfo    stage
		_computePipelineLayout,                         // VkPipelineLayout                   layout
		VK_NULL_HANDLE,									// VkPipeline                         basePipelineHandle
		-1                                              // int32_t                            basePipelineIndex
	};

	if (vkCreateComputePipelines(_logicalDevice, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &_computePipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create pipeline layout!");
	}

	//-----------------------------------------------------------------------------------------------------------------------------------------------

	// pipeline ���� �� ������ ��.
	vkDestroyShaderModule(_logicalDevice, fragShaderModule, nullptr);
	vkDestroyShaderModule(_logicalDevice, vertShaderModule, nullptr);
	//vkDestroyShaderModule(_logicalDevice, computeShaderModule, nullptr);
}

void VulkanRenderer::createRenderPass()
{
	//swap chain �̹����� �� �ϳ��� ��Ÿ���� color buffer attachment
	// color(����) buffer(����) attachment(÷�ι�)
	//https://www.notion.so/VkAttachmentDescription-Manual-Page-774a0dde223c41939b99b4b4f04349c9
	VkAttachmentDescription colorAttachment{};
	//color attachment�� format�� swap chain image���� format�� ����.
	colorAttachment.format = _swapchain.format;
	// multisampling ���� �۾� ����( ���� �� ������ 1)
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

	//attatchmnet�� �����Ͱ� ������ ��/�Ŀ� ������ �� ������ ����
	// LOAD : ���� attachment ���� / CLEAR : ����� / DONT_CARE : ���� ������ undefined
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// STORE : �������� ������ ���� �� ���� �� ���� / DONT_CARE : ������ �� undefined
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	//stencil ���� ���� (stencil ���۷� fragment ���� ���. ���⼱ ���x)
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	//Vulkan�� texture�� frame buffer�� Ư�� pixel format�� VkImage ������Ʈ�� ǥ�� ��.

	// render pass�� �����ϱ� �� ������ image layout ���� ����
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;//���� �̹����� � layout�̾��� ������� ��.
	// render pass�� ������ �� �ڵ������� ��ȯ�� layout�� ����.
	// ������ �Ŀ� swap chain�� ���� image�� presentation �� ���̱� ������ �Ʒ��� ���� ����.
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	//sub pass�� �ϳ� �̻��� attachment�� ������.
	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// ���� render pass�� �������� sub pass�� �����Ǵµ� sub pass�� 
	// ���� pass�� frame buffer ���뿡 �����ϴ� �ļ� ������ �۾��Դϴ�. (ex) post-processing)

	// ������ �ﰢ�� �ϳ� ���Ŵϱ� ���� sub pass ����.
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

	// attachment �迭�� sub pass�� ����Ͽ� VkRenderPassCreateInfo ����ü�� ä��� ���� ����!
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
		Swap Chain�� ������ surface�� ȣȯ���� ���� �� �������� ���� ���� �׸� ���� ���ǰ� �ʿ��մϴ�.
		1. Basic Surface capabilites(swap chain�� �ִ�/�ּ� �̹��� ����, �̹��� w/h �ִ� �ּ� ��)
		2. surface format(pixel format, color space)
		3. ��� ������ presentation ���
	*/
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_physicalDevice);

	_swapchain.clear();
	if (!_swapchain.handle)
		vkDestroySwapchainKHR(_logicalDevice, _swapchain.handle, nullptr);


	// 1. surface format(color depth)
	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	// 2. presentation mode(�̹����� ȭ�鿡 �������ϱ� ���� ����)
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	// 3. swap extent(swap chain�� �̹��� �ػ�)
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);


	// swap chain�� ����� �̹��� ������ ���Ѵ�. + 1 �� �ϴ� �ּ� 1�� �̻��� �̹����� ��û�ϰڴٴ� �ǹ��̴�.
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	//���� Vulkan ������Ʈ ���ʴ�� swap chain�� ����ü�� ä������.
	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = _surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	// �̹����� �����ϴ� layer�� �� (3D APP�� �������� �ʴ� �̻� �׻� 1��.)
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphic.familyIndex, indices.present.familyIndex };

	// Graphic queue family�� Presentation queue family�� �ٸ� ��� graphic queue���� 
	// swap chain �̹���, presentation queue�� �� �̹����� �����ϰ� ��.

	// Queue family���� ���Ǵ� S.C �̹��� �ڵ鸵 ����� ������.
	if (indices.graphic.familyIndex != indices.present.familyIndex)
	{
		// ������� ������ ���� ���� �̹����� ���� Queue Family���� ��� ����.
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		// �̹����� �� Queue family���� �����ϰ� �ٸ� Q.F���� ����Ϸ��� ��� ��������� ������ ����.
		// �� �ɼ��� �ֻ��� ������ ���� ��.
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	// swap chain�� trasnform( ex: 90�� ȸ��.. ���� �ø� ��). �״�� �ѰŸ� current �ϸ� ��.
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	//������ �ý��ۿ��� �ٸ� ������� ������ ���� ä�� ����� �ǰ��� ����.
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;// ���� ä�� ����
	createInfo.presentMode = presentMode;
	// ������ �ȼ��� �Ű澲�� �ʰڴٴ� ��
	createInfo.clipped = VK_TRUE;

	// ������ ������¡�� �� �� ���� swap chain�� �����ϱ� ���� �� �ʵ带 �����ؾ� ��.
	// ���������� �̰��� �ϴ� null�� ��.(���� �� �׻� ���� ����)
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	// ��� ���������� swap chain create!
	if (vkCreateSwapchainKHR(_logicalDevice, &createInfo, nullptr, &_swapchain.handle) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create swap chain!");
	}

	// ���� imageCount�� ���� �̹��� ������ ������ ��
	vkGetSwapchainImagesKHR(_logicalDevice, _swapchain.handle, &imageCount, nullptr);
	// �����̳� ũ�⸦ �����ϰ�
	_swapchain.images.resize(imageCount);
	// ���������� �̸� �ٽ� ȣ���Ͽ� �ڵ��� ���´�. �̴� �󸶵��� �� ���� ���� swapChain�� ������ �� �ֱ� �����̴�.
	vkGetSwapchainImagesKHR(_logicalDevice, _swapchain.handle, &imageCount, _swapchain.images.data());

	_swapchain.format = surfaceFormat.format;
	_swapchain.size = extent;

	//--------------------ImageView
	// swap chain ������ �°� imageViews�� ��������
	_swapchain.imageViews.resize(_swapchain.images.size());


	for (size_t i = 0; i < _swapchain.images.size(); i++)
	{
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = _swapchain.images[i];
		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = _swapchain.format;
		// color channel�� ���� �� �ֵ��� ����. (�ܻ� �ؽ�ó�� ���ٸ� ��� channel�� red�� ������ ���� ����.)
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		// �̹����� �뵵, � �κ��� �׼��� �ؾ��ϴ��� ���
		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;

		// �� ���� ������ image View create!
		if (vkCreateImageView(_logicalDevice, &createInfo, nullptr, &_swapchain.imageViews[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image views!");
		}
	}
}

void VulkanRenderer::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphic.familyIndex, indices.present.familyIndex };// , indices.compute.familyIndex};

	float queuePriority = 1.0f; // 0.0 ~ 1.0 ���̿��� ��� ����. �켱 ���� �Ҵ�.
	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		//Queue Family���� �츮�� ���ϴ� queue�� ������ ���.
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};

	// VkDeviceQueueCreateInfo, VkPhysicalDeviceFeatures�� ���� VkDeviceCreateInfo ���� ����.
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

	// �� queue family���� queue �ڵ��� ã�ƿ´�. 
	// ( logical device, queue family, queue index, queue handle ������ ������)
	vkGetDeviceQueue(_logicalDevice, indices.graphic.familyIndex, 0, &_graphicQueue);
	vkGetDeviceQueue(_logicalDevice, indices.present.familyIndex, 0, &_presentQueue);
	//vkGetDeviceQueue(_logicalDevice, indices.compute.familyIndex, 0, &_computeQueue);
}

void VulkanRenderer::createSurface()
{
	// surface ? �������� �̹����� ǥ���� ��
	//�� �÷����� �ٸ� ������ ���� Vulkan surface ����
	if (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}

SwapChainSupportDetails VulkanRenderer::querySwapChainSupport(VkPhysicalDevice physicalDevice)
{
	SwapChainSupportDetails details;

	//1. Basic Surface capabilites(swap chain�� �ִ� / �ּ� �̹��� ����, �̹��� w / h �ִ� �ּ� ��)
	// ������ Physical Device(GPU)�� ������ Surface�� ����Ͽ� �����Ǵ� capability�� ������.
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, _surface, &details.capabilities);

	//2. surface format(pixel format, color space)
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, _surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, _surface, &formatCount, details.formats.data());
	}

	//3. ��� ������ presentation ���
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
		// �̺�Ʈ �ɰ��� ����
		debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		// �ݹ��� ȣ��ǵ��� �ϴ� �̺�Ʈ ���� ����
		debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		//�ݹ� �Լ�
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
	// queue family ����Ʈ�� ����
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies)
	{
		//VK_QUEUE_GRAPHICS_BIT �� �����ϴ� �ּ� �ϳ��� queue family�� ã�ƾ� ��.
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

	// command buffer�� queue�� �ϳ��� ���������ν� ���� ��. ��� queue�� ���� ��.
	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

	// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : ���ο� command�� �ſ� ���� ���.
	// VK_COMMNAD_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : command buffer�� ���������� ���� �� �� ����.
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

void  VulkanRenderer::initWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	_window = glfwCreateWindow(WIDTH, HEIGHT, engineName.c_str(), nullptr, nullptr);

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
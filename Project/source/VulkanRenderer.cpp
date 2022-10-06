#include "VulkanRenderer.h"

#include "FileSystem.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define USE_COMPUTE 1


int texWidth, texHeight, texChannels;

void VulkanRenderer::initVulkan()
{
	createInstance();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();

	createSwapchain();
	createSwapChainImageViews();
	
	createRenderPass();

	createGraphicsDescriptorSetLayout();

	createGraphicsPipeline();

	createFramebuffers();

	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
	
	createCommandPool(indices.graphic.familyIndex, _graphic.commandPool);

	//---------------------------------------------------- load texture & create imagewView and sampler
	loadTextureImage("F:\\yuiena\\Engine\\Project\\res\\textures\\texture.png", _textureImage, _textureImageMemory);
	createTextureImageView();
	_textureSampler = createTextureSampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT,
		VK_COMPARE_OP_ALWAYS, VK_BORDER_COLOR_INT_OPAQUE_BLACK);

#if USE_COMPUTE
	VkImage			image;
	VkDeviceMemory	imageMemory;
	//---------------------------------------------------- craete Image, imageMemory
	// Image will be sampled in the fragment shader and used as storage target in the compute shader
	VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	uint32_t queueIndexCount = 0;
	uint32_t* queueIndices = nullptr;

	if (indices.graphic.familyIndex != indices.compute.familyIndex)
	{
		std::vector<uint32_t> queueFamilyIndices = { indices.graphic.familyIndex, indices.compute.familyIndex };
		sharingMode = VK_SHARING_MODE_CONCURRENT;
		queueIndexCount = 2;
		queueIndices = queueFamilyIndices.data();
	}

	createImage(_swapchain.size.width, _swapchain.size.height, _swapchain.format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory, sharingMode, queueIndexCount, queueIndices);

	//---------------------------------------------------- create Image View
	_compute.imageView = createImageView(image, _swapchain.format);

	//---------------------------------------------------- create Sampler
	_compute.sampler = createTextureSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		VK_COMPARE_OP_NEVER, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE);
#endif

	//-------------------------------------- Buffer
	createVertexBuffer();
	createIndexBuffer();

	createUniformBuffers();
	createDescriptorPool();

#if USE_COMPUTE
	// Binding 0: Input image (read-only)
	VkDescriptorSetLayoutBinding inputImageBinding = {
		0,									// shader binding
		VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,	// shader descriptorType
		1,									// descriptor Count 배열안의 몇 개의 변수가 있는지
		VK_SHADER_STAGE_COMPUTE_BIT,		// Shader stage Flag
		nullptr,							// image sampler
	};
	// Binding 1: Output image (write)
	VkDescriptorSetLayoutBinding outputImageBinding{
		1,											// shader binding
		VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,			// shader descriptorType
		1,											// descriptor Count 배열안의 몇 개의 변수가 있는지
		VK_SHADER_STAGE_COMPUTE_BIT,				// Shader stage Flag
		nullptr,									// image sampler
	};

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = { inputImageBinding, outputImageBinding };

	VkDescriptorSetLayoutCreateInfo descriptorLayout{
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,	// type
		nullptr,												// next
		0,														// DescriptorSetLayoutCreate Flags
		static_cast<uint32_t>(bindings.size()),					// binding count
		bindings.data()											// pBindings
	};

	vkCreateDescriptorSetLayout(_logicalDevice, &descriptorLayout, nullptr, &_compute.descriptorSetLayout);

	VkPipelineLayoutCreateInfo computePipelineLayoutInfo = {
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,	// VkStructureType                  sType
	nullptr,                                        // const void                     * pNext
	0,												// VkPipelineLayoutCreateFlags      flags
	1,												// uint32_t                         setLayoutCount
	&_compute.descriptorSetLayout,                  // const VkDescriptorSetLayout    * pSetLayouts
	0,												// uint32_t                         pushConstantRangeCount
	nullptr											// const VkPushConstantRange      * pPushConstantRanges
	};

	if (VK_SUCCESS != vkCreatePipelineLayout(_logicalDevice, &computePipelineLayoutInfo, nullptr, &_compute.pipelineLayout))
	{
		throw std::runtime_error("Could not create pipeline layout.");
	}

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // VkStructureType                  sType
		nullptr,                                        // const void                     * pNext
		_graphic.descriptorPool,                        // VkDescriptorPool                 descriptorPool
		1,												// uint32_t                         descriptorSetCount
		&_compute.descriptorSetLayout					// const VkDescriptorSetLayout    * pSetLayouts
	};

	_compute.descriptorSets.resize(1);// MAX_FRAMES_IN_FLIGHT);
	VkResult result = vkAllocateDescriptorSets(_logicalDevice, &descriptorSetAllocateInfo, _compute.descriptorSets.data());
	if (VK_SUCCESS != result)
	{
		throw std::runtime_error("Could not allocate descriptor sets.");
	}


	//---------------------------------------------------- Update Descriptor Set( update )
	VkDescriptorImageInfo imageInfo{ // textureComputeTarget
			_textureSampler,			// sampler
			_textureImageView,			// image view
			VK_IMAGE_LAYOUT_GENERAL		// image layout
	};

	VkDescriptorImageInfo computeImageInfo{ // textureComputeTarget
			_compute.sampler,			// sampler
			_compute.imageView,			// image view
			VK_IMAGE_LAYOUT_GENERAL		// image layout
	};

	std::vector<VkWriteDescriptorSet> descriptorWrites{
		VkWriteDescriptorSet{  // textureComputeTarget
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	// type
			nullptr,								// next 
			_compute.descriptorSets[0],				// descriptor set dst
			0,										// dstBinding
			0,										// dst Array Element
			1,										// descriptor Count
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,		// descriptor Type
			&imageInfo,						// image info
			nullptr,								// buffer info
			nullptr									// texel buffer view
		},
		VkWriteDescriptorSet{  // textureComputeTarget
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	// type
			nullptr,								// next 
			_compute.descriptorSets[0],				// descriptor set dst
			1,										// dstBinding
			0,										// dst Array Element
			1,										// descriptor Count
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,		// descriptor Type
			&computeImageInfo,						// image info
			nullptr,								// buffer info
			nullptr									// texel buffer view
		}
	};

	vkUpdateDescriptorSets(_logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
#endif 
	createDescriptorSets();

	createCommandBuffer();
	createSyncObjects();

#if USE_COMPUTE
	createComputePipeline();
#endif 
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
	createSwapChainImageViews();
	createFramebuffers();
}

void VulkanRenderer::createUniformBuffers()
{
	VkDeviceSize bufferSize = sizeof(UniformBufferObject);

	_uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	_uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _uniformBuffers[i], _uniformBuffersMemory[i]);
	}
}


void VulkanRenderer::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, _graphic.descriptorSetLayout);

	VkDescriptorSetAllocateInfo allocInfo{
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,	// type
		nullptr,										// next
		_graphic.descriptorPool,						// descriptor pool
		static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),	// descriptor set count
		layouts.data()									//set layouts
	};

	_graphic.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	if (vkAllocateDescriptorSets(_logicalDevice, &allocInfo, _graphic.descriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
	{
		VkDescriptorBufferInfo bufferInfo{
			_uniformBuffers[i],			// buffer
			0,							// offset
			sizeof(UniformBufferObject) // range
		};

		VkDescriptorImageInfo imageInfo{
			_textureSampler,							// sampler
			_textureImageView,							// image view
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL	// image layout
		};

		std::array<VkWriteDescriptorSet, 2> baseImageDescriptorWrites{
			VkWriteDescriptorSet{ // ubo
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // type
				nullptr,								// next 
				_graphic.descriptorSets[i],						// descriptor set dst
				0,										// dstBinding
				0,										// dst Array Element
				1,										// descriptor Count
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,		// descriptor Type
				nullptr,								// image info
				&bufferInfo,							// buffer info
				nullptr									// texel buffer view
			},
			VkWriteDescriptorSet{ // sampler
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// type
				nullptr,									// next 
				_graphic.descriptorSets[i],							// descriptor set dst
				1,											// dstBinding
				0,											// dst Array Element
				1,											// descriptor Count
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	// descriptor Type
				&imageInfo,									// image info
				nullptr,									// buffer info
				nullptr										// texel buffer view
			}
		};
	
		vkUpdateDescriptorSets(_logicalDevice, baseImageDescriptorWrites.size(), baseImageDescriptorWrites.data(), 0, nullptr);

#if USE_COMPUTE
		VkDescriptorImageInfo textureInfo{
			_compute.sampler,		// sampler
			_compute.imageView,		// image view
			VK_IMAGE_LAYOUT_GENERAL	// image layout
		};

		std::array<VkWriteDescriptorSet, 2> descriptorWrites{
			VkWriteDescriptorSet{ // ubo
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // type
				nullptr,								// next 
				_graphic.descriptorSets[i],				// descriptor set dst
				0,										// dstBinding
				0,										// dst Array Element
				1,										// descriptor Count
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,		// descriptor Type
				nullptr,								// image info
				&bufferInfo,							// buffer info
				nullptr									// texel buffer view
			},
			VkWriteDescriptorSet{ // sampler
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// type
				nullptr,									// next 
				_graphic.descriptorSets[i],					// descriptor set dst
				1,											// dstBinding
				0,											// dst Array Element
				1,											// descriptor Count
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	// descriptor Type
				&textureInfo,								// image info
				nullptr,									// buffer info
				nullptr										// texel buffer view
			}
		};

		vkUpdateDescriptorSets(_logicalDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
#endif
	}
}

void VulkanRenderer::createDescriptorPool()
{
	// 어떤 descriptor type을 우리의 descriptor set에 담을지 그리고 그게 얼마나 많을지를 기술
	std::vector <VkDescriptorPoolSize> poolSizes{
		// ubo
		VkDescriptorPoolSize{  
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,				
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)},	
		// sampler
		VkDescriptorPoolSize{ 
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,		
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)},
	};

#if USE_COMPUTE
	// Compute pipelines uses a storage image for image reads and writes
	poolSizes.push_back(VkDescriptorPoolSize{
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) });
#endif

	VkDescriptorPoolCreateInfo poolInfo{
		VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,		// type
		nullptr,											// next
		0,													// descriptor pool create flag
#if USE_COMPUTE
		3,													// max sets
#else
		static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),		// max sets
#endif
		static_cast<uint32_t>(poolSizes.size()),			// pool size count
		poolSizes.data()									// pool sizes data
	};
	
	if (vkCreateDescriptorPool(_logicalDevice, &poolInfo, nullptr, &_graphic.descriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor pool!");
	}
}




void VulkanRenderer::updateUniformBuffer(uint32_t currentImage)
{
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo;

	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), _swapchain.size.width / (float)_swapchain.size.height, 0.1f, 10.0f);
	// GLM은 기본적으로 OpenGL을 위해 설계됐고 vulkan에선 clip coordinate Y 좌표가 반전 시켜줘야한다.
	ubo.proj[1][1] *= -1;

	// UBO를 uniform buffer에 복사해준다. 
	void* data;
	vkMapMemory(_logicalDevice, _uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);

	memcpy(data, &ubo, sizeof(ubo));
	
	vkUnmapMemory(_logicalDevice, _uniformBuffersMemory[currentImage]);
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
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphic.pipeline);


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

	//------------- 각 프레임에 설정된 올바른 descriptor을 사용해서 shader descriptor에 실제 바인딩 하도록 함수를 업데이트 합니다
	// vkCmdDrawIndexed 호출 전에 수행해야 합니다.
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphic.pipelineLayout, 0, 1, &_graphic.descriptorSets[_currentFrame], 0, nullptr);

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
			1,//VK_REMAINING_MIP_LEVELS,                  // uint32_t                   level Count
			0,                                        // uint32_t                   baseArrayLayer
			1, //VK_REMAINING_ARRAY_LAYERS                 // uint32_t                   layer Count
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
	
	

#if USE_COMPUTE
	//------------------------------------------- Submit compute Pipieline
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };

	// queue submit(제출) 및 동기화
	// ------------------- Submit compute commands
	VkSubmitInfo computeSubmitInfo{
		VK_STRUCTURE_TYPE_SUBMIT_INFO,		// type
		nullptr,							// next
		1,									// wait SemaphoreCount
		&_graphic.semaphore,				// wait semaphore
		waitStages,							// wait dst stage mask
		1,//static_cast<uint32_t>(_compute.commandBuffers.size()), // command buffer count
		&_compute.commandBuffers[0],		// command buffer
		1,									// signal semaphore count
		&_compute.semaphore					// signal semaphore
	};

	//위에서 셋팅했던 것들로 graphics queue에 command buffer 제출 가능.
	if (vkQueueSubmit(_compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer!");
	}
#endif
	vkWaitForFences(_logicalDevice, 1, &_inFlightFences[_currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	//1. swap chain으로 부터 image 획득
	VkResult result = vkAcquireNextImageKHR(_logicalDevice, _swapchain.handle, UINT64_MAX, _presentCompeleteSemaphores[_currentFrame], VK_NULL_HANDLE, &imageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) 
	{
		recreateSwapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) 
	{
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	updateUniformBuffer(_currentFrame);

	vkResetFences(_logicalDevice, 1, &_inFlightFences[_currentFrame]);

	vkResetCommandBuffer(_graphic.commandBuffers[_currentFrame], /*VkCommandBufferResetFlagBits*/ 0);

	recordCommandBuffer(_graphic.commandBuffers[_currentFrame], imageIndex);
	

#if USE_COMPUTE
	//------------------------------------------- Submit graphics commands
	//semaphore들이 실행이 시작되기 전에 기다려아 하는지, pipeline의 stage(들)을 기다려야하는지 지정.
	VkSemaphore waitSemaphores[] = { _compute.semaphore };// , _presentCompeleteSemaphores[_currentFrame]};
	//실행이 완료 됐을때 signal보낼 semaphore.
	VkSemaphore signalSemaphores[] = { _graphic.semaphore };// , _renderCompeletedSemaphores[_currentFrame]};
	VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSubmitInfo submitInfo{
		VK_STRUCTURE_TYPE_SUBMIT_INFO,			// type
		nullptr,								// next
		1,										// wait SemaphoreCount
		waitSemaphores,							// wait semaphore
		graphicsWaitStageMasks,					// wait dst stage mask
		1,										// command buffer count
		&_graphic.commandBuffers[_currentFrame],// command buffer
		1,										// signal semaphore count
		signalSemaphores						// signal semaphore
	};


	//위에서 셋팅했던 것들로 graphics queue에 command buffer 제출 가능.
	if (vkQueueSubmit(_graphic.queue, 1, &submitInfo, _inFlightFences[_currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer!");
	}

#else
	//semaphore들이 실행이 시작되기 전에 기다려아 하는지, pipeline의 stage(들)을 기다려야하는지 지정.
	VkSemaphore waitSemaphores[] = {  _presentCompeleteSemaphores[_currentFrame] };
	//실행이 완료 됐을때 signal보낼 semaphore.
	VkSemaphore signalSemaphores[] = { _renderCompeletedSemaphores[_currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSubmitInfo submitInfo{
		VK_STRUCTURE_TYPE_SUBMIT_INFO,
		nullptr,
		1,
		waitSemaphores,
		waitStages,
		1,
		&_graphic.commandBuffers[_currentFrame],
		1,
		signalSemaphores
	};

	if (vkQueueSubmit(_graphic.queue, 1, &submitInfo, _inFlightFences[_currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer!");
	}

#endif

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


void VulkanRenderer::cleanupSwapchain()
{
	for (auto framebuffer : _swapchain.framebuffers)
	{
		vkDestroyFramebuffer(_logicalDevice, framebuffer, nullptr);
	}

	for (auto imageView : _swapchain.imageViews)
	{
		vkDestroyImageView(_logicalDevice, imageView, nullptr);
	}

	vkDestroySwapchainKHR(_logicalDevice, _swapchain.handle, nullptr);
}

void VulkanRenderer::cleanup()
{
	cleanupSwapchain();

	//------------------- Compute
	vkDestroyPipeline(_logicalDevice, _compute.pipeline, nullptr);

	vkDestroyPipelineLayout(_logicalDevice, _compute.pipelineLayout, nullptr);

	vkDestroyDescriptorSetLayout(_logicalDevice, _compute.descriptorSetLayout, nullptr);

	vkDestroySemaphore(_logicalDevice, _compute.semaphore, nullptr);

	vkDestroyCommandPool(_logicalDevice, _compute.commandPool, nullptr);
	//-------------------

	vkDestroyPipeline(_logicalDevice, _graphic.pipeline, nullptr);
	vkDestroyPipelineLayout(_logicalDevice, _graphic.pipelineLayout, nullptr);
	vkDestroyRenderPass(_logicalDevice, _renderPass, nullptr);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyBuffer(_logicalDevice, _uniformBuffers[i], nullptr);
		vkFreeMemory(_logicalDevice, _uniformBuffersMemory[i], nullptr);
	}

	vkDestroyDescriptorPool(_logicalDevice, _graphic.descriptorPool, nullptr);

	vkDestroySampler(_logicalDevice, _textureSampler, nullptr);
	vkDestroyImageView(_logicalDevice, _textureImageView, nullptr);
	vkDestroyImage(_logicalDevice, _textureImage, nullptr);
	vkFreeMemory(_logicalDevice, _textureImageMemory, nullptr);

	vkDestroyDescriptorSetLayout(_logicalDevice, _graphic.descriptorSetLayout, nullptr);

	vkDestroyBuffer(_logicalDevice, _indexBuffer, nullptr);
	vkFreeMemory(_logicalDevice, _indexBufferMemory, nullptr);

	vkDestroyBuffer(_logicalDevice, _vertexBuffer, nullptr);
	vkFreeMemory(_logicalDevice, _vertexBufferMemory, nullptr);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(_logicalDevice, _renderCompeletedSemaphores[i], nullptr);
		vkDestroySemaphore(_logicalDevice, _presentCompeleteSemaphores[i], nullptr);
		
		vkDestroyFence(_logicalDevice, _inFlightFences[i], nullptr);
	}
	vkDestroySemaphore(_logicalDevice, _graphic.semaphore, nullptr);
	vkDestroyCommandPool(_logicalDevice, _graphic.commandPool, nullptr);

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
	_presentCompeleteSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	_renderCompeletedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	//_graphic.semaphore.resize(MAX_FRAMES_IN_FLIGHT);
	_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		if (vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_presentCompeleteSemaphores[i]) != VK_SUCCESS ||
			vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_renderCompeletedSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(_logicalDevice, &fenceInfo, nullptr, &_inFlightFences[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create synchronization objects for a frame!");
		}
	}

	if (vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_graphic.semaphore) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create synchronization objects for a frame!");
	}

}

void VulkanRenderer::createSwapChainImageViews()
{
	//--------------------ImageView
	// swap chain 개수에 맞게 imageViews도 리사이즈
	_swapchain.imageViews.resize(_swapchain.images.size());

	for (size_t i = 0; i < _swapchain.images.size(); i++)
	{
		_swapchain.imageViews[i] = createImageView(_swapchain.images[i], _swapchain.format);

	}
}

void VulkanRenderer::createCommandBuffer()
{
	_graphic.commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	// VK_COMMAND_BUFFER_LEVEL_PRIMARY : 실행을 위해 queue에 제출될 수 있지만 다른 command buffer에서 호출 x
	// VK_COMMAND_BUFFER_LEVEL_SECONDARY : 직접 실행 x, primary command buffer에서 호출 o
	VkCommandBufferAllocateInfo allocInfo = {
	  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,   // VkStructureType          sType
	  nullptr,                                          // const void             * pNext
	  _graphic.commandPool,                                     // VkCommandPool            commandPool
	  VK_COMMAND_BUFFER_LEVEL_PRIMARY,                  // VkCommandBufferLevel     level
	 (uint32_t)_graphic.commandBuffers.size()												// uint32_t                 commandBufferCount
	};

	if (vkAllocateCommandBuffers(_logicalDevice, &allocInfo, &_graphic.commandBuffers[_currentFrame]) != VK_SUCCESS)
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
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	endSingleTimeCommands(commandBuffer);
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


void VulkanRenderer::createGraphicsDescriptorSetLayout()
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

	VkDescriptorSetLayoutBinding samplerLayoutBinding{
		1,											// shader binding
		VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	// shader descriptorType
		1,											// descriptor Count 배열안의 몇 개의 변수가 있는지
		VK_SHADER_STAGE_FRAGMENT_BIT,				// Shader stage Flag
		nullptr,									// image sampler
	};

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

	VkDescriptorSetLayoutCreateInfo layoutInfo{
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,	// type
		nullptr,												// next
		0,														// DescriptorSetLayoutCreate Flags
		static_cast<uint32_t>(bindings.size()),					// binding count
		bindings.data()											// pBindings
	};

	// 모든 descriptor binding은 _descriptorSetLayout( VkDescriptorSetLayout )으로 결합된다.
	if (vkCreateDescriptorSetLayout(_logicalDevice, &layoutInfo, nullptr, &_graphic.descriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor set layout!");
	}

}

// Build a single command buffer containing the compute dispatch commands
void VulkanRenderer::dispatchCompute(VkCommandBuffer command_buffer, uint32_t width, uint32_t height, uint32_t depth)
{
	// 현재 사용 중인지 아닌지 확인하기 위해 pipeline변경 후 command buffer를 다시 빌드하는 경우 queue를 다시 flush합니다.
	vkQueueWaitIdle(_compute.queue);

	// command buffer 기록 시작.
	VkCommandBufferBeginInfo cmdBufInfo{};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	// command buffer 기록 시작
	vkBeginCommandBuffer(_compute.commandBuffers[0], &cmdBufInfo);

	vkCmdBindPipeline(_compute.commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, _compute.pipeline);
	vkCmdBindDescriptorSets(_compute.commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, _compute.pipelineLayout, 0, 1, &_compute.descriptorSets[_currentFrame], 0, 0);

	vkCmdDispatch(_compute.commandBuffers[0], width, height, depth);

	// command buffer 기록 끝
	vkEndCommandBuffer(_compute.commandBuffers[0]);
}

void VulkanRenderer::createComputePipeline()
{
	//---------------------------------------------------- create Compute Pipeline

	auto computeShaderCode = readFile("F:\\yuiena\\Engine\\Project\\res\\shader\\emboss.comp.spv");

	VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

	//---------------------------------------------------- create pipeline layout


	//---------------------------------------------------- Create Compute Pipeline

	VkPipelineShaderStageCreateInfo computeShaderStageInfo{
		VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,	// type
		nullptr,												// next
		0,														// flag
		VK_SHADER_STAGE_COMPUTE_BIT,							// stage
		computeShaderModule,									// loyout
		"main",													// base pipeline handle
		nullptr													// base pipeline index
	};

	VkComputePipelineCreateInfo computePipelineCreateInfo = {
	  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,   // VkStructureType                    sType
	  nullptr,                                          // const void                       * pNext
	  0,												// VkPipelineCreateFlags              flags
	  computeShaderStageInfo,							// VkPipelineShaderStageCreateInfo    stage
	  _compute.pipelineLayout,							// VkPipelineLayout                   layout
	  VK_NULL_HANDLE,									// VkPipeline                         basePipelineHandle
	  -1                                                // int32_t                            basePipelineIndex
	};

	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VkResult result = vkCreatePipelineCache(_logicalDevice, &pipelineCacheCreateInfo, nullptr, &_pipelineCache);
	if (VK_SUCCESS != result)
	{
		throw std::runtime_error("failed to create VkPipelineCacheCreateInfo!");
	}

	result = vkCreateComputePipelines(_logicalDevice, _pipelineCache, 1, &computePipelineCreateInfo, nullptr, &_compute.pipeline);
	if (VK_SUCCESS != result)
	{
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	//---------------------------------------------------- command pool create / Separate command pool as queue family for compute may be different than graphics
	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

	VkCommandPoolCreateInfo poolInfo = {
	  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,		// VkStructureType              sType
	  nullptr,											// const void                 * pNext
	  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,  // VkCommandPoolCreateFlags     flags
	  indices.compute.familyIndex						// uint32_t                     queueFamilyIndex
	};

	result = vkCreateCommandPool(_logicalDevice, &poolInfo, nullptr, &_compute.commandPool);
	if (VK_SUCCESS != result)
	{
		std::cout << "Could not create command pool." << std::endl;
	}

	//---------------------------------------------------- Create a command buffer for compute operations
	VkCommandBufferAllocateInfo allocInfo
	{
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // type
		nullptr,										// next
		_compute.commandPool,									// command pool
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,				// buffer level
		1												// command buffer count
	};

	_compute.commandBuffers.resize(1);
	vkAllocateCommandBuffers(_logicalDevice, &allocInfo, &_compute.commandBuffers[0]);


	//---------------------------------------------------- Semaphore for compute & graphics sync
	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	if (vkCreateSemaphore(_logicalDevice, &semaphoreInfo, nullptr, &_compute.semaphore) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create synchronization objects for a frame!");
	}
	
	//---------------------------------------------------- build compute command bufefr
	dispatchCompute(_compute.commandBuffers[0], texWidth / 16, texHeight, 1);

	//~

	vkDestroyShaderModule(_logicalDevice, computeShaderModule, nullptr);
}



void VulkanRenderer::createGraphicsPipeline()
{
	//----------------------------------------------------------------------------------------------------------
	auto vertShaderCode = readFile("F:\\yuiena\\Engine\\Project\\res\\shader\\vert_image.spv");
	auto fragShaderCode = readFile("F:\\yuiena\\Engine\\Project\\res\\shader\\frag_image.spv");


	// 파이프라인에 코드를 전달하기 위해 VkShaderModule 오브젝트로 랩핑 해야함.
	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
	
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

	

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	/* ☆Input Assetmbler		- raw vertex 데이터 수집
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
	// proj Y-flip때문에 vertex는 CW order가 아니라 CCW order로 그려지고 있다. 이는 backface culling이 시작되어 gemoetry가 그려지는 것을 막기 때문에 아래의 플래그를 바꿔준다.
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
	// ( 어떤 descriptor가 shader에 사용될지 알려주기 위해 pipeline 생성 동안 descriptor set layout 지정 )
	// Descriptor set with storage image
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // VkStructureType                  sType
		nullptr,                                        // const void                     * pNext
		0,                                              // VkPipelineLayoutCreateFlags      flags
		1,												// uint32_t                         setLayoutCount
		&_graphic.descriptorSetLayout,					// const VkDescriptorSetLayout    * pSetLayouts
		0,												// uint32_t                         pushConstantRangeCount
		nullptr											// const VkPushConstantRange      * pPushConstantRanges
	};

	if (vkCreatePipelineLayout(_logicalDevice, &pipelineLayoutInfo, nullptr, &_graphic.pipelineLayout) != VK_SUCCESS)
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
	pipelineInfo.layout = _graphic.pipelineLayout;
	pipelineInfo.renderPass = _renderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	// graphics pipeline create!
	if (vkCreateGraphicsPipelines(_logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_graphic.pipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create graphics pipeline!");
	}


	// pipeline 생성 후 지워야 함.
	vkDestroyShaderModule(_logicalDevice, fragShaderModule, nullptr);
	vkDestroyShaderModule(_logicalDevice, vertShaderModule, nullptr);
	
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

	
#if USE_COMPUTE
	uint32_t queueFamilyIndices[] = { static_cast<uint32_t>(indices.graphic.familyIndex), indices.present.familyIndex, indices.compute.familyIndex };
#else
	uint32_t queueFamilyIndices[] = { indices.graphic.familyIndex, indices.present.familyIndex };
#endif 

	// Graphic queue family와 Presentation queue family가 다른 경우 graphic queue에서 
	// swap chain 이미지, presentation queue에 그 이미지를 제출하게 됨.

	// Queue family간에 사용되는 S.C 이미지 핸들링 방법을 지정함.
	if (indices.graphic.familyIndex != indices.present.familyIndex)
	{
		// 명시적인 소유권 전송 없이 이미지는 여러 Queue Family에서 사용 가능.
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		if (indices.graphic.familyIndex != indices.compute.familyIndex)
		{
			createInfo.queueFamilyIndexCount = 3;
		}
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		if (indices.graphic.familyIndex != indices.compute.familyIndex)
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
	}

	// swap chain의 trasnform( ex: 90도 회전.. 수평 플립 등). 그대로 둘거면 current 하면 됨.
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	//윈도우 시스템에서 다른 윈도우와 블렌딩시 알파 채널 사용할 건가를 지정.
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;// 알파 채널 무시
	createInfo.presentMode = presentMode;
	// 가려진 픽셀을 신경쓰지 않겠다는 뜻
	createInfo.clipped = VK_TRUE;


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


void VulkanRenderer::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
#if USE_COMPUTE
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphic.familyIndex, indices.present.familyIndex, indices.compute.familyIndex};
#else
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphic.familyIndex, indices.present.familyIndex };// , indices.compute.familyIndex};
#endif
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
	deviceFeatures.samplerAnisotropy = VK_TRUE;

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
	vkGetDeviceQueue(_logicalDevice, indices.graphic.familyIndex, 0, &_graphic.queue);
	vkGetDeviceQueue(_logicalDevice, indices.present.familyIndex, 0, &_presentQueue);
#if USE_COMPUTE
	vkGetDeviceQueue(_logicalDevice, indices.compute.familyIndex, 0, &_compute.queue);
#endif
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

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);

	return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
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


void VulkanRenderer::createCommandPool(uint32_t familyIndex, VkCommandPool& commandPool)
{
	if (_physicalDevice == NULL)
	{
		throw std::runtime_error("_physicalDevice is NULL!");
	}

	// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 새로운 command가 매우 자주 기록.
	// VK_COMMNAD_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : command buffer가 개별적으로 재기록 될 수 있음.
	VkCommandPoolCreateInfo poolInfo = {
	  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,		// VkStructureType              sType
	  nullptr,											// const void                 * pNext
	  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,  // VkCommandPoolCreateFlags     flags
	  familyIndex										// uint32_t                     queueFamilyIndex
	};

	VkResult result = vkCreateCommandPool(_logicalDevice, &poolInfo, nullptr, &commandPool);
	if (VK_SUCCESS != result)
	{
		std::cout << "Could not create command pool." << std::endl;
		return;
	}
}

VkSampler VulkanRenderer::createTextureSampler(VkSamplerAddressMode u, VkSamplerAddressMode v, VkSamplerAddressMode w, VkCompareOp compareOp, VkBorderColor borderColor)
{
	VkSampler sampler;

	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(_physicalDevice, &properties);

	VkSamplerCreateInfo samplerInfo{
		VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,	// type
		nullptr,								// next
		0,										// Sampler create flag
		VK_FILTER_LINEAR,						// mag filter - 확대, 축소된 texel을 보간하는 방법을 지정
		VK_FILTER_LINEAR,						// min filter- 확대, 축소된 texel을 보간하는 방법을 지정
		VK_SAMPLER_MIPMAP_MODE_LINEAR,			// mipmapMode
		u,										// addressModeU
		v,										// addressModeV
		w,										// addressModeW
		0.0f,									// mip Lod Bias
		VK_TRUE,								// anisotropy Enable
		properties.limits.maxSamplerAnisotropy, // max Anisotropy
		VK_FALSE,								// compare Enable
		compareOp,								// compare OP
		0.0f,									// min Lod
		0.0f,									// max Lod
		borderColor,							// border color
		VK_FALSE								// unnormalized coordinate - image의 texel을 지정하는데 사용할 좌표계 지정
	};

	if (vkCreateSampler(_logicalDevice, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create texture sampler!");
	}

	return sampler;
}

void VulkanRenderer::createTextureImageView()
{
	_textureImageView = createImageView(_textureImage, VK_FORMAT_R8G8B8A8_SRGB);
}

VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format)
{
	VkImageViewCreateInfo viewInfo{
		VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,	// type
		nullptr,									// nullptr
		0,			// image view create flag
		image,										// image
		VK_IMAGE_VIEW_TYPE_2D,						// image view type
		format,										// format
		VkComponentMapping{							// component
			VK_COMPONENT_SWIZZLE_R,
			VK_COMPONENT_SWIZZLE_G,
			VK_COMPONENT_SWIZZLE_B,
			VK_COMPONENT_SWIZZLE_A},
		VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,	// aspect mask
			0,							// base mip level
			1,							// level count
			0,							// base array layer
			1}							// layer count
	};

	VkImageView imageView;
	if (vkCreateImageView(_logicalDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create texture image view!");
	}

	return imageView;
}

void VulkanRenderer::loadTextureImage(const std::string& path, VkImage& targetImage, VkDeviceMemory targetTextureMemory)
{
	// STBI_rgb_alpha : alpha 채널이 없어도 이미지를 강제 로드 하므로 향후 다른 텍스처와의 일관성에 좋다.
	stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels) 
	{
		throw std::runtime_error("failed to load texture image!");
	}

	// image buffer를 전송할 수 있는 staging buffer, memory 생성
	createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _stagingBuffer, _stagingBufferMemory);

	// staging buffer memory에 pixel 복사.
	void* data;
	vkMapMemory(_logicalDevice, _stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(_logicalDevice, _stagingBufferMemory);

	// pixel free
	stbi_image_free(pixels);

	//VK_SHARING_MODE_EXCLUSIVE,			// sharing Mode
	//	0,									// queue family index count
	//	nullptr,							// queue family indices
	//	VK_IMAGE_LAYOUT_UNDEFINED			// initial layout

	// 이미지 생성
	createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, targetImage, targetTextureMemory,
		VK_SHARING_MODE_EXCLUSIVE, 0, nullptr);

	// image layout 전환
	transitionImageLayout(targetImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	// staging buffer에 image copy
	copyBufferToImage(_stagingBuffer, targetImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

	//shader texture image에서 sampling을 시작하려면 shader access를 준비하기 위해 마지막 전환필요
	transitionImageLayout(targetImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(_logicalDevice, _stagingBuffer, nullptr);
	vkFreeMemory(_logicalDevice, _stagingBufferMemory, nullptr);
}

void VulkanRenderer::Test()
{

}

void VulkanRenderer::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferImageCopy region
	{
		0,						// buffer offset
		0,						// buffer row length
		0,						// buffer image height
		VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,	//aspec mask
			0,							// mip Level
			0,							// base array layer
			1},							// layer count
		VkOffset3D{0, 0, 0},
		VkExtent3D{width, height, 1},
		
	};

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(commandBuffer);
}

void VulkanRenderer::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	// layout 전환을 수행하는 가장 일반적인 방법 중 하나는 image memory barrier를 사용하는 것이다.
	// 이와 같은 pipeline barrier는 일반적으로 buffer에서 읽기 전에 쓰기가 완료되도록 하는 것과 같이 리소스에 대한 액세스를 동기화 하는데 
	// 사용되지만 사용될 때 image layout을 전환하고 queue family 소유권을 이전하는데도 사용할 수 있다.
	VkImageMemoryBarrier barrier{
		VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // type
		nullptr,								// next
		0,										// src access mask
		0,										// dst access mask
		oldLayout,								// image layout old
		newLayout,								// image layout new
		VK_QUEUE_FAMILY_IGNORED,				// src queue family index (queue family index를 이전하지 않을 시 VK_QUEUE_FAMILY_IGNORED)
		VK_QUEUE_FAMILY_IGNORED,				// dst queue family index
		image,									// image
		VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,	// aspect mask
			0,							// base mip level
			1,							// level count
			0,							// base array layer
			1}							// layer count
	};
	
	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	// transition layout 기반으로 설정해야하는 두가지 전환
	// Undefined  : 전송대상 - 아무 것도 기다릴 필요가 없는 전송 쓰기
	// Transfer destination  : Shader 읽기 - Shader 읽기는 fragment shader에서 대기해야합니다. 바로 여기에 texture를 사용할 것이기 떄문입니다.
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) 
	{
		// VK_ACCESS_HOST_WRITE_BIT 는 처음에 암시적인 동기화를 초래함으로 주의해야하며 처음엔 0으로 설정.
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) 
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else 
	{
		throw std::invalid_argument("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(
		commandBuffer,		// command buffer
		sourceStage,		// src stage mask
		destinationStage,	// dst stage mask
		0,					// dependency flag
		0,					// memoryh barrier count
		nullptr,			// memory barriers
		0,					// buffer memory barrier count
		nullptr,			// buffer memory barriers
		1,					// image memory barrier count
		&barrier			// image memory barriers
	);

	endSingleTimeCommands(commandBuffer);
}

void VulkanRenderer::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, 
	VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, VkSharingMode sharingMode, uint32_t queueIndexCount, uint32_t* queueIndices)
{
	VkImageCreateInfo imageInfo
	{
		VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,// type
		nullptr,							// next
		0,									// Image Create Flag
		VK_IMAGE_TYPE_2D,					// Image Type
		format,								// Format
		VkExtent3D{width, height, 1},		// exten3D
		1,									// mipLevels
		1,									// array Layers
		VK_SAMPLE_COUNT_1_BIT,				// sample Count Flag Bits
		tiling,								// Image Tiling
		usage,								// Image Usage Flag
		sharingMode,						// sharing Mode
		queueIndexCount,					// queue family index count
		queueIndices,						// queue family indices
		VK_IMAGE_LAYOUT_UNDEFINED			// initial layout
	};

	

	if (vkCreateImage(_logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) 
	{
		throw std::runtime_error("failed to create image!");
	}

	// buffer에 메모리를 할당하는 것과 정확히 같은 방식으로 동작. vkGetBufferMemoryRequirement대신 vkGetImageMemoryRequirements를 사용한다.
	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(_logicalDevice, image, &memRequirements);

	VkMemoryAllocateInfo allocInfo
	{
		VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,						// type
		nullptr,													// next
		memRequirements.size,										// allocation size
		findMemoryType(memRequirements.memoryTypeBits, properties)	// memory Type index
	};

	if (vkAllocateMemory(_logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate image memory!");
	}

	vkBindImageMemory(_logicalDevice, image, imageMemory, 0);
}

VkCommandBuffer VulkanRenderer::beginSingleTimeCommands()
{
	VkCommandBufferAllocateInfo allocInfo
	{
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // type
		nullptr,										// next
		_graphic.commandPool,							// command pool
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,				// buffer level
		1												// command buffer count
	};

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(_logicalDevice, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,	// type
		nullptr,										// next
		VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,	// buffer usage flag
		nullptr											// buffer inheritance info
	};

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void VulkanRenderer::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	submitInfo.pSignalSemaphores = &_graphic.semaphore;


	vkQueueSubmit(_graphic.queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(_graphic.queue);

	vkFreeCommandBuffers(_logicalDevice, _graphic.commandPool, 1, &commandBuffer);
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
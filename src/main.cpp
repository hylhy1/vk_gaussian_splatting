/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gaussian_splatting.h>
#include <GLFW/glfw3.h>
#include <render_system.h>
#include <vulkan_context.h>
#include <nvvk/check_error.hpp>

using namespace vk_gaussian_splatting;

// create, setup and run an nvapp::Application
// with a GaussianSplatting element.
int main(int argc, char** argv)
{
  nvutils::Logger::getInstance().breakOnError(false);
  //nvutils::Logger::getInstance().setLogLevel(nvutils::Logger::LogLevel::eDEBUG);

  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser(nvutils::getExecutablePath().stem().string(), {".txt"});
  nvutils::ParameterSequencer::InitInfo sequencerInfo{// sequencer always requires a parser and registry
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry};

  nvvk::Context                vkContext;  // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;    // Information to create the Vulkan context
  RenderSystem                renderSystem;  // 自定义渲染系统
  bool                         benchmarkMode = false;

  /////////////////////////////////
  // Parse the command line to get the application creation information
  // those parameter will have no effect if changed via benchmark script
  // see GaussianSplatting constructor for other options
  parameterRegistry.add({"verbose", "Verbose output of the Vulkan context"}, &vkSetup.verbose);
  parameterRegistry.add({"validation", "Enable validation layers"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"benchmark", "Enable benchmarking, prevents async loadings and turns off vsync"}, &benchmarkMode);
  parameterRegistry.add({"forcegpu", "Force the use of a specific GPU by probviding its ID"}, &vkSetup.forceGPU);

  registerCommandLineParameters(&parameterRegistry);

  // The GaussianSplattingUI includes the core GaussianSplatting class by inheritance
  auto gaussianSplatting = std::make_shared<GaussianSplatting>(nullptr, &parameterRegistry);
  //sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);
  // extends reporting output with memory consumption information
  // After the creation of the elements we have more parameters in the registry than before (from gaussianSplatting).
  // Therefore add the entire registry to the commandline parser again, to add new ones.
  parameterParser.add(parameterRegistry);
  // commandline parsing
  parameterParser.parse(argc, argv);
  storeDefaultParameters();
  // set more verbose for benchmark usage later on
  parameterParser.setVerbose(true);

  // this element requires sequencerInfo that is potentially updated by parameterParser
  //auto elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);

  /////////////////////////////////
  // Vulkan creation context information
  vkSetup.enableAllFeatures = true;

  // - Instance extensions
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // - Device extensions
  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  static VkPhysicalDeviceMeshShaderFeaturesEXT meshFeaturesEXT = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
  };

  static VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
  };
  vkSetup.deviceExtensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);  // for vk_radix_sort (vrdx)
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshFeaturesEXT, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &fragFeaturesKHR, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeaturesKHR, true);

  // Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature, true);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature, false);  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeatures);

  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures = {
      .sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
      .rayTracingInvocationReorder = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &serFeatures, false);


  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  

  // Setting up the validation layers
  nvvk::ValidationSettings vvlInfo{};

  vvlInfo.validate_core = false;

  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain(); 

  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }  
  // 手动创建命令池
  VkCommandPoolCreateInfo cmdPoolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = vkContext.getQueueInfos()[0].familyIndex,
  };
  VkCommandPool commandPool;
  NVVK_CHECK(vkCreateCommandPool(vkContext.getDevice(), &cmdPoolInfo, nullptr, &commandPool));

  // 手动创建描述符池
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
  };
  VkDescriptorPoolCreateInfo descriptorPoolInfo{
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets       = 1000,
      .poolSizeCount = static_cast<uint32_t>(std::size(poolSizes)),
      .pPoolSizes    = poolSizes,
  };
  VkDescriptorPool descriptorPool;
  NVVK_CHECK(vkCreateDescriptorPool(vkContext.getDevice(), &descriptorPoolInfo, nullptr, &descriptorPool));

  // 创建并初始化VulkanContext
  vk_gaussian_splatting::VulkanContext vkCtx;
  vkCtx.init(vkContext.getInstance(), vkContext.getPhysicalDevice(), vkContext.getDevice(),
             commandPool, descriptorPool,
             vkContext.getQueueInfos()[0], {1280, 720}, true);

  // 手动调用onAttach，传入VulkanContext
  gaussianSplatting->onAttach(&vkCtx);

  // 初始化GLFW
  if(!glfwInit())
  {
    LOGE("Failed to initialize GLFW\n");
    return 1;
  }
  
  LOGI("Creating GLFW window: %dx%d\n", vkCtx.getViewportSize().width, vkCtx.getViewportSize().height);

  // 手动创建GLFW窗口
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
  
  GLFWwindow* window = glfwCreateWindow(vkCtx.getViewportSize().width, vkCtx.getViewportSize().height, 
                                        "vulkan 3dgs", nullptr, nullptr);
  if(!window)
  {
    const char* error;
    glfwGetError(&error);
    LOGE("Failed to create GLFW window: %s\n", error ? error : "unknown error");
    glfwTerminate();
    return 1;
  }

  // 初始化自定义渲染系统
  if(!renderSystem.init(vkContext.getInstance(), vkContext.getPhysicalDevice(), vkContext.getDevice(),
                       vkContext.getQueueInfos()[0], window))
  {
    LOGE("Failed to initialize render system\n");
    glfwDestroyWindow(window);
    return 1;
  }

  // 创建交换链
  VkExtent2D windowSize{vkCtx.getViewportSize().width, vkCtx.getViewportSize().height};
  if(!renderSystem.createSwapchain(windowSize, vkCtx.getVSync()))
  {
    LOGE("Failed to create swapchain\n");
    glfwDestroyWindow(window);
    return 1;
  }

  // 使用自定义渲染循环
  LOGI("Running application with custom render system\n");
  
  // 初始化GBuffer大小
  VkCommandBuffer initCmd = renderSystem.beginFrame();
  if(initCmd != VK_NULL_HANDLE)
  {
    gaussianSplatting->onResize(initCmd, vkCtx.getViewportSize());
    renderSystem.endFrame(initCmd);
    renderSystem.presentFrame();
  }
  // 窗口大小变化回调
  VkExtent2D currentWindowSize = vkCtx.getViewportSize();
  glfwSetWindowUserPointer(window, &currentWindowSize);
  glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) {
    VkExtent2D* size = static_cast<VkExtent2D*>(glfwGetWindowUserPointer(window));
    size->width = width;
    size->height = height;
  });

  // 主渲染循环
  while(!glfwWindowShouldClose(window))
  {
    // 窗口系统事件
    glfwPollEvents();
    
    // 检查窗口大小变化
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  VkExtent2D newWindowSize{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  if(newWindowSize.width != currentWindowSize.width || newWindowSize.height != currentWindowSize.height)
  {
    // 重新初始化交换链，这会修改newWindowSize为实际的交换链大小
    renderSystem.reinitSwapchain(newWindowSize, vkCtx.getVSync());
    
    // 更新当前窗口大小
    currentWindowSize = newWindowSize;
    vkCtx.setViewportSize(newWindowSize);
    
    // 更新GBuffer大小
    VkCommandBuffer resizeCmd = renderSystem.beginFrame();
    if(resizeCmd != VK_NULL_HANDLE)
    {
      gaussianSplatting->onResize(resizeCmd, newWindowSize);
      renderSystem.endFrame(resizeCmd);
      renderSystem.presentFrame();
    }
    // 跳过这一帧，因为交换链刚重新创建
    continue;
  }
    
    // 跳过最小化时的渲染
    if(glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    
    // 开始帧
    VkCommandBuffer cmd = renderSystem.beginFrame();
    if(cmd == VK_NULL_HANDLE)
    {
      continue;
    }
    
    // 渲染GaussianSplatting到GBuffer
    gaussianSplatting->onPreRender();
    gaussianSplatting->onRender(cmd);
    // 获取GBuffer和交换链的大小
    VkExtent2D gBufferSize = gaussianSplatting->getGBufferSize();
    VkExtent2D swapchainSize = renderSystem.getSwapchainSize();
    
    // 转换GBuffer颜色图像布局为传输源
    VkImageMemoryBarrier gBufferBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = gaussianSplatting->getGBufferColorImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &gBufferBarrier);
    
    // 转换交换链图像布局为传输目标（从PRESENT_SRC_KHR）
    VkImageMemoryBarrier swapchainBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = renderSystem.getSwapchainImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchainBarrier);
    
    // 将GBuffer内容复制到交换链
    VkImageBlit blit{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {static_cast<int32_t>(gBufferSize.width), static_cast<int32_t>(gBufferSize.height), 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{0, 0, 0}, {static_cast<int32_t>(swapchainSize.width), static_cast<int32_t>(swapchainSize.height), 1}},
    };
    vkCmdBlitImage(cmd, gaussianSplatting->getGBufferColorImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  renderSystem.getSwapchainImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    
    // 转换交换链图像布局为呈现源
    swapchainBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapchainBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapchainBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapchainBarrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchainBarrier);
    
    // 转换GBuffer颜色图像布局回GENERAL
    gBufferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    gBufferBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    gBufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    gBufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &gBufferBarrier);
    
    // 结束帧
    renderSystem.endFrame(cmd);
    
    // 呈现帧
    renderSystem.presentFrame();
  }
  
  // 等待所有帧完成
  renderSystem.waitForFrameCompletion();
  
  // 等待设备空闲，确保所有GPU操作都已完成
  vkDeviceWaitIdle(vkContext.getDevice());
  
  // 调用onDetach清理GaussianSplatting
  gaussianSplatting->onDetach();
  
  // 销毁交换链
  renderSystem.destroySwapchain();
  
  // 销毁渲染系统
  renderSystem.deinit();
  
  // 销毁窗口
  glfwDestroyWindow(window);
  
  // 终止GLFW
  glfwTerminate();
  
  // 清理命令池和描述符池
  vkDestroyCommandPool(vkContext.getDevice(), commandPool, nullptr);
  vkDestroyDescriptorPool(vkContext.getDevice(), descriptorPool, nullptr);
  
  // 清理VulkanContext
  vkCtx.deinit();
  
  // Cleanup VulkanContext
  vkContext.deinit();

  return 0;
}

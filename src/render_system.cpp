/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "render_system.h"
#include <GLFW/glfw3.h>
#include <volk.h>
#include <nvvk/commands.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/barriers.hpp>

namespace vk_gaussian_splatting {

RenderSystem::RenderSystem()
{
}

RenderSystem::~RenderSystem()
{
  deinit();
}

bool RenderSystem::init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                      const nvvk::QueueInfo& queueInfo, GLFWwindow* window)
{
  m_instance = instance;
  m_physicalDevice = physicalDevice;
  m_device = device;
  m_queue = queueInfo;
  m_window = window;

  // 创建命令池
  VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = queueInfo.familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool));

  return true;
}

void RenderSystem::deinit()
{
  destroySwapchain();

  if(m_commandPool != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    m_commandPool = VK_NULL_HANDLE;
  }

  m_instance = VK_NULL_HANDLE;
  m_physicalDevice = VK_NULL_HANDLE;
  m_device = VK_NULL_HANDLE;
  m_queue = {};
  m_window = nullptr;
  m_surface = VK_NULL_HANDLE;
}

bool RenderSystem::createSwapchain(VkExtent2D windowSize, bool vsync)
{
  // 获取surface（如果不存在则创建）
  if(m_surface == VK_NULL_HANDLE)
  {
    glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
  }

  // 初始化交换链
  nvvk::Swapchain::InitInfo swapchainInfo{
      .physicalDevice = m_physicalDevice,
      .device         = m_device,
      .queue          = m_queue,
      .surface        = m_surface,
      .cmdPool        = m_commandPool,
  };

  VkResult result = m_swapchain.init(swapchainInfo);
  if(result != VK_SUCCESS)
  {
    return false;
  }

  result = m_swapchain.initResources(windowSize, vsync);
  if(result != VK_SUCCESS)
  {
    m_swapchain.deinit();
    return false;
  }

  m_maxFramesInFlight = m_swapchain.getMaxFramesInFlight();
  m_currentFrameIndex = 0;

  // 创建fence用于帧同步
  m_inFlightFences.resize(m_maxFramesInFlight);
  VkFenceCreateInfo fenceInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };
  for(uint32_t i = 0; i < m_maxFramesInFlight; ++i)
  {
    NVVK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]));
  }
  
  // 创建命令缓冲区池
  m_commandBuffers.resize(m_maxFramesInFlight);
  VkCommandBufferAllocateInfo allocInfo{
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = m_commandPool,
      .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = m_maxFramesInFlight,
  };
  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()));

  return true;
}

void RenderSystem::destroySwapchain()
{
  // 等待队列空闲，确保所有semaphore都已完成
  if(m_queue.queue != VK_NULL_HANDLE)
  {
    vkQueueWaitIdle(m_queue.queue);
  }
  
  // 清理fence
  for(VkFence fence : m_inFlightFences)
  {
    if(fence != VK_NULL_HANDLE)
    {
      vkDestroyFence(m_device, fence, nullptr);
    }
  }
  m_inFlightFences.clear();
  
  // 清理命令缓冲区
  if(!m_commandBuffers.empty())
  {
    vkFreeCommandBuffers(m_device, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()), m_commandBuffers.data());
    m_commandBuffers.clear();
  }

  // 销毁swapchain（deinit会调用deinitResources）
  m_swapchain.deinit();
  
  // 销毁surface
  if(m_surface != VK_NULL_HANDLE)
  {
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
  }
}

void RenderSystem::reinitSwapchain(VkExtent2D& windowSize, bool vsync)
{
  // 等待队列空闲，确保所有semaphore都已完成
  if(m_queue.queue != VK_NULL_HANDLE)
  {
    vkQueueWaitIdle(m_queue.queue);
  }
  
  // 重新初始化swapchain资源
  VkResult result = m_swapchain.reinitResources(windowSize, vsync);
  if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    return;
  }
}

VkCommandBuffer RenderSystem::beginFrame()
{
  // 检查fence是否已创建
  if(m_inFlightFences.empty())
  {
    return VK_NULL_HANDLE;
  }
  
  // 确保currentFrameIndex在有效范围内
  if(m_currentFrameIndex >= m_inFlightFences.size())
  {
    m_currentFrameIndex = 0;
  }
  
  // 等待当前帧的fence
  vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrameIndex], VK_TRUE, UINT64_MAX);
  vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrameIndex]);

  // 获取下一个交换链图像
  VkResult result = m_swapchain.acquireNextImage(m_device);
  if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    return VK_NULL_HANDLE;
  }

  // 获取当前帧的命令缓冲区
  VkCommandBuffer cmd = m_commandBuffers[m_currentFrameIndex];

  // 重置命令缓冲区
  NVVK_CHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
  
  // 不再需要转换到COLOR_ATTACHMENT_OPTIMAL，因为我们使用blit直接复制到交换链
  
  return cmd;
}

void RenderSystem::endFrame(VkCommandBuffer cmd)
{
  NVVK_CHECK(vkEndCommandBuffer(cmd));

  // 检查fence是否已创建
  if(m_inFlightFences.empty())
  {
    return;
  }
  
  // 确保currentFrameIndex在有效范围内
  if(m_currentFrameIndex >= m_inFlightFences.size())
  {
    m_currentFrameIndex = 0;
  }

  // 提交命令缓冲
  VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSemaphore imageAvailable = m_swapchain.getImageAvailableSemaphore();
  VkSemaphore renderFinished = m_swapchain.getRenderFinishedSemaphore();
  VkSubmitInfo submitInfo{
      .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount   = 1,
      .pWaitSemaphores    = &imageAvailable,
      .pWaitDstStageMask  = &waitStages,
      .commandBufferCount   = 1,
      .pCommandBuffers      = &cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores  = &renderFinished,
  };

  NVVK_CHECK(vkQueueSubmit(m_queue.queue, 1, &submitInfo, m_inFlightFences[m_currentFrameIndex]));
}

void RenderSystem::presentFrame()
{
  m_swapchain.presentFrame(m_queue.queue);
  m_currentFrameIndex = (m_currentFrameIndex + 1) % m_maxFramesInFlight;
}

VkImage RenderSystem::getSwapchainImage() const
{
  return m_swapchain.getImage();
}

VkImageView RenderSystem::getSwapchainImageView() const
{
  return m_swapchain.getImageView();
}

VkExtent2D RenderSystem::getSwapchainSize() const
{
  // 返回窗口大小，这里简化处理
  int width, height;
  glfwGetWindowSize(m_window, &width, &height);
  return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

void RenderSystem::beginRenderingToSwapchain(VkCommandBuffer cmd) const
{
  beginRenderingToSwapchain(cmd, false);
}

void RenderSystem::beginRenderingToSwapchain(VkCommandBuffer cmd, bool load) const
{
  const VkRenderingAttachmentInfo colorAttachment{
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_swapchain.getImageView(),
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp      = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = {{{0.0f, 0.0f, 0.0f, 1.0f}}},
  };

  VkExtent2D windowSize = getSwapchainSize();
  const VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea           = {{0, 0}, windowSize},
      .layerCount           = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments    = &colorAttachment,
  };

  // 不需要布局转换，因为图像已经在VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  vkCmdBeginRendering(cmd, &renderingInfo);
}

void RenderSystem::endRenderingToSwapchain(VkCommandBuffer cmd)
{
  vkCmdEndRendering(cmd);
  nvvk::cmdImageMemoryBarrier(cmd, {m_swapchain.getImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});
}

void RenderSystem::waitForFrameCompletion() const
{
  for(const VkFence& fence : m_inFlightFences)
  {
    if(fence != VK_NULL_HANDLE)
    {
      vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);
    }
  }
}

}  // namespace vk_gaussian_splatting

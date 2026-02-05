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

#ifndef _RENDER_SYSTEM_H_
#define _RENDER_SYSTEM_H_

#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>
#include <nvvk/swapchain.hpp>
#include <nvvk/queue.hpp>
#include <vector>
#include <memory>

namespace vk_gaussian_splatting {

class RenderSystem
{
public:
  RenderSystem();
  ~RenderSystem();

  bool init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
             const nvvk::QueueInfo& queueInfo, GLFWwindow* window);
  void deinit();

  bool createSwapchain(VkExtent2D windowSize, bool vsync);
  void destroySwapchain();
  void reinitSwapchain(VkExtent2D& windowSize, bool vsync);

  VkCommandBuffer beginFrame();
  void endFrame(VkCommandBuffer cmd);
  void presentFrame();

  VkImage getSwapchainImage() const;
  VkImageView getSwapchainImageView() const;
  VkExtent2D getSwapchainSize() const;

  void beginRenderingToSwapchain(VkCommandBuffer cmd) const;
  void beginRenderingToSwapchain(VkCommandBuffer cmd, bool load) const;
  void endRenderingToSwapchain(VkCommandBuffer cmd);

  void waitForFrameCompletion() const;

private:
  VkInstance                   m_instance{VK_NULL_HANDLE};
  VkPhysicalDevice             m_physicalDevice{VK_NULL_HANDLE};
  VkDevice                     m_device{VK_NULL_HANDLE};
  nvvk::QueueInfo             m_queue{};
  GLFWwindow*                  m_window{nullptr};
  VkSurfaceKHR                 m_surface{VK_NULL_HANDLE};

  nvvk::Swapchain m_swapchain;
  VkCommandPool    m_commandPool{VK_NULL_HANDLE};

  uint32_t m_currentFrameIndex{0};
  uint32_t m_maxFramesInFlight{2};

  std::vector<VkFence> m_inFlightFences;
  std::vector<VkCommandBuffer> m_commandBuffers;
};

}  // namespace vk_gaussian_splatting

#endif  // _RENDER_SYSTEM_H_

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

#ifndef _VULKAN_CONTEXT_H_
#define _VULKAN_CONTEXT_H_

#include <vulkan/vulkan_core.h>
#include <nvvk/queue.hpp>

namespace vk_gaussian_splatting {

struct VulkanContext
{
  VkInstance                   instance{VK_NULL_HANDLE};
  VkPhysicalDevice             physicalDevice{VK_NULL_HANDLE};
  VkDevice                     device{VK_NULL_HANDLE};
  VkCommandPool                commandPool{VK_NULL_HANDLE};
  VkDescriptorPool             textureDescriptorPool{VK_NULL_HANDLE};
  nvvk::QueueInfo             queue{};
  VkExtent2D                  viewportSize{0, 0};
  bool                        vSync{false};

  void init(VkInstance inst, VkPhysicalDevice physDevice, VkDevice dev, 
            VkCommandPool cmdPool, VkDescriptorPool texDescPool, 
            const nvvk::QueueInfo& queueInfo, const VkExtent2D& viewSize, bool vsync = false)
  {
    instance = inst;
    physicalDevice = physDevice;
    device = dev;
    commandPool = cmdPool;
    textureDescriptorPool = texDescPool;
    queue = queueInfo;
    viewportSize = viewSize;
    vSync = vsync;
  }

  void deinit()
  {
    instance = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    commandPool = VK_NULL_HANDLE;
    textureDescriptorPool = VK_NULL_HANDLE;
    viewportSize = {0, 0};
    vSync = false;
  }

  VkInstance getInstance() const { return instance; }
  VkDevice getDevice() const { return device; }
  VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
  const nvvk::QueueInfo& getQueue(uint32_t index) const { return queue; }
  VkCommandPool getCommandPool() const { return commandPool; }
  VkDescriptorPool getTextureDescriptorPool() const { return textureDescriptorPool; }
  const VkExtent2D& getViewportSize() const { return viewportSize; }
  void setViewportSize(const VkExtent2D& size) { viewportSize = size; }
  bool getVSync() const { return vSync; }
  void setVSync(bool vsync) { vSync = vsync; }

  VkCommandBuffer createTempCmdBuffer() const;
  void submitAndWaitTempCmdBuffer(VkCommandBuffer cmd) const;

};

}  // namespace vk_gaussian_splatting

#endif  // _VULKAN_CONTEXT_H_

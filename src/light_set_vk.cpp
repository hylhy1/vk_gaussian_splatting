/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "light_set_vk.h"

#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

void LightSetVk::init(nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader)
{
  m_alloc    = alloc;
  m_uploader = uploader;
  lights.resize(MAX_LIGHTS);
  // create the buffer
  NVVK_CHECK(m_alloc->createBuffer(lightsBuffer, lights.size() * sizeof(shaderio::LightSource), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(lightsBuffer.buffer);
  // buffer will be uploaded when updateBuffer is called with a command buffer
};

// free the vulkan buffer and clear the light set
void LightSetVk::deinit()
{
  m_alloc->destroyBuffer(lightsBuffer);
  // reset default light
  numLights = 1;
  lights[0] = shaderio::LightSource();
};

void LightSetVk::updateBuffer(VkCommandBuffer cmd)
{
  // Upload the lights buffer
  NVVK_CHECK(m_uploader->appendBuffer(lightsBuffer, 0, std::span(lights)));
  m_uploader->cmdUploadAppended(cmd);
  m_uploader->releaseStaging();
}

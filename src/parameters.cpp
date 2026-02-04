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

#include "parameters.h"

namespace vk_gaussian_splatting {

// no reset function on purpose
SceneParameters prmScene{};

VramDataParameters    prmData{};
RtxVramDataParameters prmRtxData{};

// no reset function on purpose
uint32_t            prmSelectedPipeline = PIPELINE_VERT;
shaderio::FrameInfo prmFrame{};
RenderParameters    prmRender{};
RasterParameters    prmRaster{};
RtxParameters       prmRtx{};

// Storage for respective default values

static VramDataParameters    prmDataDefault{};
static RtxVramDataParameters prmRtxDataDefault{};

static shaderio::FrameInfo prmFrameDefault{};
static RenderParameters    prmRenderDefault{};
static RasterParameters    prmRasterDefault{};
static RtxParameters       prmRtxDefault{};

void storeDefaultParameters()
{
  prmDataDefault    = prmData;
  prmRtxDataDefault = prmRtxData;

  prmFrameDefault  = prmFrame;
  prmRenderDefault = prmRender;
  prmRasterDefault = prmRaster;
  prmRtxDefault    = prmRtx;
}

void resetDataParameters()
{
  prmData = prmDataDefault;
}
void resetRtxDataParameters()
{
  prmRtxData = prmRtxDataDefault;
}
void resetFrameParameters()
{
  prmFrame = prmFrameDefault;
}
void resetRenderParameters()
{
  prmRender = prmRenderDefault;
}
void resetRasterParameters()
{
  prmRaster = prmRasterDefault;
}
void resetRtxParameters()
{
  prmRtx = prmRtxDefault;
}

void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry)
{
  
}

}  // namespace vk_gaussian_splatting

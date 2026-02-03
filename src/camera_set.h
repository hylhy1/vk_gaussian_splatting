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

#ifndef _CAMERA_SET_H_
#define _CAMERA_SET_H_

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <tinygltf/json.hpp>

#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/acceleration_structures.hpp>

#include <nvutils/camera_manipulator.hpp>

#include "shaderio.h"

namespace vk_gaussian_splatting {

typedef nvutils::CameraManipulator::Camera NvutilCamera;

struct Camera
{
  int model = CAMERA_PINHOLE;

  glm::vec3 eye = glm::vec3(0.0F, 0.0F, 2.0F);  // camera position
  glm::vec3 ctr = glm::vec3(0, 0, 0);           // center of rotation (look at point) for interaction
  glm::vec3 up  = glm::vec3(0, 1, 0);           // up vector

  float     fov  = 60.0f;            // field of view
  glm::vec2 clip = {0.1f, 2000.0f};  // znear, zfar

  bool  dofEnabled = false;
  float focusDist  = 1.3f;    // focus distance to compute depth of field (defocus effect)
  float aperture   = 0.001f;  // aperture distance to compute depth of field, 0 does no DOF effect

  bool operator==(const Camera& cam) const
  {
    return model == cam.model && eye == cam.eye && ctr == cam.ctr && up == cam.up && fov == cam.fov && clip == cam.clip;
  }
};

class CameraSet
{

public:
  void init(nvutils::CameraManipulator* cameraManip) { m_cameraManip = cameraManip; }

  void deinit()
  {
    m_camera = Camera();
    m_presets.clear();
  }

  // any modification of the camera must be followed by a setCamera for the change to take place
  Camera getCamera() { return applyNvutilCamera(m_cameraManip->getCamera(), m_camera); }

  void setCamera(const Camera& camera, bool instantSet = true)
  {
    m_camera = camera;
    m_cameraManip->setCamera(toNvutilCamera(camera), instantSet);
  };

  void setHomePreset(const Camera& camera)
  {
    if(m_presets.empty())
      m_presets.resize(1);
    m_presets[0] = camera;
  }

  // return the number of cameras in the set
  uint64_t size() const { return m_presets.size(); }

  // store the current camera parameter in a new preset entry
  // if a preset with same attributes already exists, no additional
  // preset is created and index of existing one is returned
  uint64_t createPreset(const Camera& camera)
  {
    bool     unique = true;
    uint64_t i      = 0;
    for(; i < m_presets.size(); ++i)
    {
      if(m_presets[i] == camera)
      {
        unique = false;
        break;
      }
    }
    if(unique)
    {
      m_presets.emplace_back(camera);
      return m_presets.size() - 1;
    }
    else
    {
      return i;
    }
  }

  // store the current camera parameter in a new camera entry
  // is a camera with same attributes already exists, no additional
  // camera is created and index of existing one is returned
  uint64_t storeCurrentCamera(void) { return createPreset(getCamera()); }

  // load a camera entry and set
  bool loadPreset(uint64_t index, bool instantSet = true)
  {
    if(index >= m_presets.size())
      return false;
    setCamera(m_presets[index], instantSet);
    return true;
  }

  // remove a camera from the set
  // default one (index 0) is forbiden
  bool erasePreset(uint64_t index)
  {
    if(m_presets.size() == 1 || index == 0 || index >= m_presets.size())
      return false;

    for(uint64_t i = index; i < m_presets.size() - 1; ++i)
    {
      m_presets[i] = m_presets[i + 1];
    }
    m_presets.resize(m_presets.size() - 1);
    return true;
  }

  // access the camera source of given index
  Camera getPreset(uint64_t index) const
  {
    assert(index < m_presets.size());
    return m_presets[index];
  }

  // access the camera source of given index
  bool setPreset(uint64_t index, const Camera& camera)
  {
    if(m_presets.size() == 1 || index == 0 || index >= m_presets.size())
      return false;
    m_presets[index] = camera;
    return true;
  }

private:
  Camera                      m_camera;       // active camera
  std::vector<Camera>         m_presets;      // set of camera presets
  nvutils::CameraManipulator* m_cameraManip;  // camer manipulatror (navigation/animation)

  // preserves the additional fields of camera
  Camera& applyNvutilCamera(const NvutilCamera nvuCam, Camera& camera)
  {
    camera.eye  = nvuCam.eye;
    camera.ctr  = nvuCam.ctr;
    camera.up   = nvuCam.up;
    camera.fov  = nvuCam.fov;
    camera.clip = nvuCam.clip;
    return camera;
  }

  Camera toCamera(const NvutilCamera nvuCam)
  {
    // other fields of result are set to default
    return {
        .eye  = nvuCam.eye,
        .ctr  = nvuCam.ctr,
        .up   = nvuCam.up,
        .fov  = nvuCam.fov,
        .clip = nvuCam.clip,
    };
  }

  NvutilCamera toNvutilCamera(const Camera& camera)
  {
    // other fields from camera are "lost"
    return {
        .eye  = camera.eye,
        .ctr  = camera.ctr,
        .up   = camera.up,
        .fov  = camera.fov,
        .clip = camera.clip,
    };
  }
};

}  // namespace vk_gaussian_splatting

#endif
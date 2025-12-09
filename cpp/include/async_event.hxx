/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 */

#pragma once

namespace nvgraph {

  class AsyncEvent {
    public:
      AsyncEvent() : async_event(NULL) { }
      AsyncEvent(int size) : async_event(NULL) { hipEventCreate(&async_event); }
      ~AsyncEvent() { if (async_event != NULL) hipEventDestroy(async_event); }

      void create() { hipEventCreate(&async_event); }
      void record(hipStream_t s=0) {
        if (async_event == NULL)  
          hipEventCreate(&async_event); // check if we haven't created the event yet
        hipEventRecord(async_event,s);
      }
      void sync() {
        hipEventSynchronize(async_event);
      }
    private:
      hipEvent_t async_event;
  };

}


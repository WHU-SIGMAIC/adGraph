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

#include "nvgraph_error.hxx"

namespace nvgraph
{


  void nvgraph_default_output(const char *msg, int length) {
#if defined(DEBUG) || defined(VERBOSE_DIAG)
    printf("%s", msg);
#endif
  }

  NVGRAPH_output_callback nvgraph_output = nvgraph_default_output;
  NVGRAPH_output_callback error_output = nvgraph_default_output;
  //NVGRAPH_output_callback nvgraph_distributed_output = nvgraph_default_output;*/

  // Timer 
  struct cuda_timer::event_pair
  {
    hipEvent_t start;
    hipEvent_t end;
  };
  cuda_timer::cuda_timer(): p(new event_pair()) { }
  
  void cuda_timer::start()
  {
    hipEventCreate(&p->start);
    hipEventCreate(&p->end);
    hipEventRecord(p->start, 0);
    cudaCheckError();
  }
  float cuda_timer::stop()
  {
    hipEventRecord(p->end, 0);
    hipEventSynchronize(p->end);
    float elapsed_time;
    hipEventElapsedTime(&elapsed_time, p->start, p->end);
    hipEventDestroy(p->start);
    hipEventDestroy(p->end);
    cudaCheckError();
    return elapsed_time;
  }

} // end namespace nvgraph


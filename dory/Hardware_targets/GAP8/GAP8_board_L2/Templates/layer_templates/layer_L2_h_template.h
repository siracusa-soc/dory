/*
 * layer_template_h.h
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

% if sdk == 'gap_sdk':
#include "pulp.h"
% endif
#include "dory.h"
<<<<<<<< HEAD:Hardware-targets/nnx/Templates/layer_templates/layer_L2_h_template.h
#include "pulp_nnx.h"
========
#include "pulp_nn_kernels.h"
>>>>>>>> origin/master:dory/Hardware_targets/GAP8/GAP8_board_L2/Templates/layer_templates/layer_L2_h_template.h

void  ${func_name}(
  void *args
);

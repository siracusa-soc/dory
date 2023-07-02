/*
<<<<<<<< HEAD:Hardware-targets/Diana/Templates/layer_templates/layer_L2_h_template.h
 * layer_template_h.h
========
 * network.h
>>>>>>>> origin/master:dory/Hardware_targets/GAP8/GAP8_board_L2/Templates/input_h_template.h
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

#ifndef __INPUT_H__
#define __INPUT_H__
#include "pmsis.h"
% if sdk == 'gap_sdk':
L2_DATA uint8_t ${prefix}L2_input_h[${dimension}] = {
% elif sdk == 'pulp-sdk':
PI_L2 uint8_t ${prefix}L2_input_h[${dimension}] = {
% endif
${input_values}};
#endif

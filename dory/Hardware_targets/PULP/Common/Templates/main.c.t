/*
 * test_template.c
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
<%
l3_supported = DORY_HW_graph[0].HW_description['memory']['levels'] > 2
n_inputs = DORY_HW_graph[0].n_test_inputs
single_input = n_inputs==1
%>\
% if not l3_supported:
#include "input.h"
% else:
#include "mem.h"
% endif
#include "network.h"

#include "pmsis.h"

% if verbose:
#define VERBOSE 1
% endif

% if sdk == 'pulp-sdk':
unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies) {
  return 0;
}
% endif


int main () {

% if sdk == 'pulp-sdk':
  #if __PLATFORM__ == ARCHI_PLATFORM_FPGA
    *(int*)(ICACHE_PREFETCH) = 0xFFFF;
  #endif
% endif

/*
    Opening of Filesystem and Ram
*/
% if l3_supported:
  mem_init();
  network_initialize();
  % endif
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(${l2_buffer_size});
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)l2_buffer, l2_buffer?"Ok":"Failed");
#endif
  size_t l2_input_size = ${int(DORY_HW_graph[0].tiling_dimensions["L2"]["input_activation_memory"])};
  size_t input_size = 1000000;
  % if l3_supported:

  void *ram_input = ram_malloc(input_size);
  % endif
% if not single_input:
  for (int exec = 0; exec < ${n_inputs}; exec++) {
    % if l3_supported:
  load_file_to_ram(ram_input, Input_names[exec]);
    % endif
% elif l3_supported:

  load_file_to_ram(ram_input, "inputs.hex");
% endif

  % if l3_supported:
  ram_read(l2_buffer, ram_input, l2_input_size);
  % endif

      network_run(l2_buffer, ${l2_buffer_size}, l2_buffer, ${"0" if single_input else "exec"}${f", L2_input_h{' + exec * l2_input_size' if not single_input else ''}" if not l3_supported else ""});

  % if not single_input:
  }
  % endif
  % if l3_supported:
  ram_free(ram_input, input_size);
  network_terminate();
  % endif
  pi_l2_free(l2_buffer, ${l2_buffer_size});
}

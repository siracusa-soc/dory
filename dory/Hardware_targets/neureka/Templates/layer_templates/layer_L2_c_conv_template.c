/*
 * layer_template_nnx.c
 * Francesco Conti <f.conti@unibo.it>
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright (C) 2018-2022 University of Bologna
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

#include "${func_name}.h"
#include "pulp_nnx.h"
#include "${prefix}network.h"
#include "dory_get_tile.h"
#include "dory_dma.h"

#ifdef GVSOC_LOGGING
#define GVSOC_LOG_LEVEL 3
#include "pulp_nnx_util.h"
#endif GVSOC_LOGGING

#ifndef PRINT_DMA
#define PRINT_DMA
void static print_dma(DMA_copy* dma){
  printf("\n[dma] ext:%p, loc:%p, n_2d:%u, s_2d:%u, n_1d:%u, s_1d:%u, l_1d:%u\n", dma->ext, dma->loc, dma->number_of_2d_copies, dma->stride_2d, dma->number_of_1d_copies, dma->stride_1d, dma->length_1d_copy); 
}
#endif

#ifdef DEBUG_DMA_COPY
#define dory_dma_memcpy_async(dma)                                                                                             \
  do                                                                                                                           \
  {                                                                                                                            \
    print_dma(dma);							\
    dory_dma_memcpy_async(dma);                                                                                                \
  } while (0)
#endif

% if ULTRA_VERBOSE:
// #define VERBOSE_PRINT(...) printf(__VA_ARGS__)
#define VERBOSE_PRINT(...)
% endif

#define MIN(a,b) (a < b ? a : b)

#ifdef RUNTIMEMEASUREMENT
static uint32_t cycles, tot_cycles, tile_cycles = 0;
#endif

// DMA_Y_CONTEXT_SIZE
// At least NNX_CONTEXT_SIZE + 1 DMA_copy_y configurations are needed because output
// is always 2 phases late, so there are 2 configurations for previous stages
// and 1 for the current. It can be done differently but it sacrifices code
// readability which was prioritiesed at the moment.
// Size of 4 has been assigned to have index calculation done with only masking.
#define DMA_Y_CONTEXT_SIZE 4
#define DMA_Y_MASK 0x3
#define DMA_Y_INDEX(n) (n & DMA_Y_MASK)

#define DMA_RESHUFFLE 1

static int x_t_start, x_t_end = 0;
static uint32_t startCycles, endCycles, STARTED = 0;

static int increment_i_dma_y(int i) {
  return (i + 1) != DMA_Y_CONTEXT_SIZE ? i + 1 : 0; 
}

static void contract_strided_output(int i_store_y, DMA_copy* DMA_copy_y, uint32_t* wEffY, uint32_t dma_channel){
  
  uint8_t* y_tile_ptr = DMA_copy_y[DMA_Y_INDEX(i_store_y)].loc;
  uint32_t blockwidth = DMA_copy_y[DMA_Y_INDEX(i_store_y)].length_1d_copy;

  uint32_t cp_y_tile_size_h = DMA_copy_y[DMA_Y_INDEX(i_store_y)].number_of_2d_copies;
  uint32_t cp_y_tile_size_w = DMA_copy_y[DMA_Y_INDEX(i_store_y)].number_of_1d_copies;
  uint32_t cp_y_tile_size_w_eff = wEffY[DMA_Y_INDEX(i_store_y)];

#if DMA_RESHUFFLE
  
  DMA_copy reshuffle_copy;
  reshuffle_copy.hwc_to_chw = 0;
  reshuffle_copy.dir = 1;
  reshuffle_copy.tid = dma_channel;
  
  reshuffle_copy.loc = y_tile_ptr;
  reshuffle_copy.ext = y_tile_ptr;
  reshuffle_copy.length_1d_copy = blockwidth;
  reshuffle_copy.number_of_2d_copies = cp_y_tile_size_h;
  reshuffle_copy.number_of_1d_copies = cp_y_tile_size_w;
  reshuffle_copy.stride_2d = ${stride}*wEffY[DMA_Y_INDEX(i_store_y)]*blockwidth;
  reshuffle_copy.stride_1d = ${stride}*blockwidth;
  
  dory_dma_barrier(&reshuffle_copy);  
  dory_dma_memcpy_async(&reshuffle_copy);
  dory_dma_barrier(&reshuffle_copy);

 #else
  
  for (int idx_h = 0; idx_h < cp_y_tile_size_h; idx_h++){
    for (int idx_w = 0; idx_w < cp_y_tile_size_w; idx_w++){
      memcpy(y_tile_ptr + (idx_h*cp_y_tile_size_w + idx_w)*blockwidth, y_tile_ptr + (idx_h*cp_y_tile_size_w_eff*${stride} + idx_w*${stride})*blockwidth, blockwidth);
    }
  }

#endif
  
}

void ${func_name}(
  void *args
) {
  /////////////
  // Logging //
  /////////////
/* #ifdef GVSOC_LOGGING */
/*   nnx_activate_gvsoc_logging(GVSOC_LOG_LEVEL); */
/* #endif */

  //////////////////////////
  // Arguments assignment //
  //////////////////////////

  // Keep the same interface between L2 and L3 memory
  layer_args_t *layer_args = (layer_args_t *) args;
  const unsigned int l2_x = layer_args->L2_input;
  const unsigned int l2_y = layer_args->L2_output;
  const unsigned int l2_W = layer_args->L2_weights;
% if FLAG_BATCHNORM == 1:
  %if use_wmem:
  const unsigned int l2_scale = ${func_name}_k;
  const unsigned int l2_bias  = ${func_name}_l;
  %else:
  const unsigned int l2_scale = l2_W + ${l2_k_offset - l2_W_offset};
  const unsigned int l2_bias  = l2_W + ${l2_lambda_offset - l2_W_offset};
  %endif
% endif
  const unsigned int l1_buffer = layer_args->L1_buffer;
  const unsigned int out_shift = layer_args->out_shift;

  int pad_offset_h, pad_offset_w;
  
  /////////////////////
  // DMA declaration //
  /////////////////////

  uint32_t dory_dma_channel = dory_dma_allocate();
  DMA_copy DMA_copy_W, DMA_copy_x;
% if FLAG_BATCHNORM == 1:
  DMA_copy DMA_copy_k, DMA_copy_lambda;
% endif
  DMA_copy DMA_copy_y[DMA_Y_CONTEXT_SIZE];
  uint32_t wEffY[DMA_Y_CONTEXT_SIZE];
  int dma_copy_y_job_ids[DMA_Y_CONTEXT_SIZE];

  //////////////////
  // DMA defaults //
  //////////////////

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${l1_x_dma_stride_2d};
  DMA_copy_x.stride_1d = ${l1_x_dma_stride_1d};
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dory_dma_channel;
  
  DMA_copy_W.hwc_to_chw = 0;
  DMA_copy_W.number_of_2d_copies = 1;
  DMA_copy_W.stride_2d = 0;
  DMA_copy_W.number_of_1d_copies = 1;
  DMA_copy_W.stride_1d = 0;
  DMA_copy_W.dir = 1;
  DMA_copy_W.tid = dory_dma_channel;

% if FLAG_BATCHNORM == 1:
  DMA_copy_k.hwc_to_chw = 0;
  DMA_copy_k.stride_2d = 0;
  DMA_copy_k.number_of_2d_copies = 1;
  DMA_copy_k.stride_1d = 0;
  DMA_copy_k.number_of_1d_copies = 1;
  DMA_copy_k.dir = 1;
  DMA_copy_k.tid = dory_dma_channel;

  DMA_copy_lambda.hwc_to_chw = 0;
  DMA_copy_lambda.stride_2d = 0;
  DMA_copy_lambda.number_of_2d_copies = 1;
  DMA_copy_lambda.stride_1d = 0;
  DMA_copy_lambda.number_of_1d_copies = 1;
  DMA_copy_lambda.dir = 1;
  DMA_copy_lambda.tid = dory_dma_channel;
  
% endif
  
  for (int i = 0; i < DMA_Y_CONTEXT_SIZE; i++) {
    DMA_copy_y[i].hwc_to_chw = 0;
    DMA_copy_y[i].stride_2d = ${l1_y_dma_stride_2d};
    DMA_copy_y[i].stride_1d = ${l1_y_dma_stride_1d};
    DMA_copy_y[i].dir = 0;
    DMA_copy_y[i].tid = dory_dma_channel;
  }

% if has_bias == 1:
  DMA_copy DMA_copy_bias;
  DMA_copy_bias.hwc_to_chw = 0;
  DMA_copy_bias.stride_2d = 0;
  DMA_copy_bias.stride_1d = 0;
  DMA_copy_bias.dir = 1;
  DMA_copy_bias.tid = dory_dma_channel;

% endif

  //////////////////////////
  // Variable declaration //
  //////////////////////////

  int y_tile_size_h = ${y_tile_size_h};
  int y_tile_size_w = ${y_tile_size_w};
    %if stride > 1:
  int y_tile_size_h_eff = ${y_tile_size_h * stride - ((x_tile_size_h + padding_top) % stride)};
  int y_tile_size_w_eff = ${y_tile_size_w * stride - ((x_tile_size_w + padding_left) % stride)};
  %else:
     int y_tile_size_h_eff = ${y_tile_size_h};
  int y_tile_size_w_eff = ${y_tile_size_w};
  %endif
  int y_length_nof_byte = ${y_tile_size_nof_byte};

  int x_tile_size_h = ${x_tile_size_h};
  int x_tile_size_w = ${x_tile_size_w};
  int x_length_nif_byte = ${x_tile_size_nif_byte};

  int W_tile_size_nof = ${W_tile_size_nof};
  int W_tile_size_nif = ${W_tile_size_nif};
  int W_tile_ko_len = ${l1_W_tile_ko_len};

  // Tile loop indices
  int i_nof = 0, i_nif = 0, i_h = 0, i_w = 0;

  // Double buffer pointer indices
  int i_db_x = 0, i_db_y = 0, i_db_w = 0;

  // Store iterator
  int i_store_y = 0;

  // Load flags (first tile must be loaded)
  int is_load_w = 1, is_load_x = 1;

  ////////////////////////
  // Double buffer init //
  ////////////////////////

  const int l1_buffer_x = l1_buffer + ${l1_x_offset};
  const int l1_buffer_y = l1_buffer + ${l1_y_offset};
  const int l1_buffer_w = l1_buffer + ${l1_W_offset};
% if FLAG_BATCHNORM:
  const int l1_buffer_scale = l1_buffer + ${l1_k_offset};
  const int l1_buffer_bias = l1_buffer + ${l1_lambda_offset};
% endif

  const struct {
% if FLAG_BATCHNORM == 1:
    int scale;
    int bias;
% endif
    int x;
    int y;
    int w;
  } db[2] = {
    {
% if FLAG_BATCHNORM == 1:
      .scale = l1_buffer_scale,
      .bias = l1_buffer_bias,
% endif
      .x = l1_buffer_x,
      .y = l1_buffer_y,
      .w = l1_buffer_w
    },
    {
% if FLAG_BATCHNORM == 1:
      .scale = l1_buffer_scale + ${l1_k_tile_size},
      .bias = l1_buffer_bias + ${l1_lambda_tile_size},
% endif
      .x = l1_buffer_x + ${l1_x_tile_size},
      .y = l1_buffer_y + ${l1_y_tile_size},
      .w = l1_buffer_w + ${l1_W_tile_size}
    }
  };

  //////////////////////
  // Accelerator init //
  //////////////////////

  NEUREKA_SETPRIORITY_NEUREKA();
  NEUREKA_RESET_MAXSTALL();
  NEUREKA_SET_MAXSTALL(8);
  
  nnx_soft_clear();

  ///////////////////////
  // NNX task defaults //
  ///////////////////////

  enum nnx_task_e {
    NNX_TASK_BODY,
    NNX_TASK_REMAINDER,
    NNX_TASK_COUNT
  };

  nnx_task_t  nnx_tasks[NNX_TASK_COUNT];
  nnx_task_t *nnx_task_to_offload;


  nnx_weights_t nnx_weights = {
    .data = db[i_db_w].w,
    .height = ${fs1},
    .width = ${fs2},
    .depth = ${W_tile_size_nif},
    .n_weights = ${W_tile_size_nof},
    .bitwidth = ${W_data_size},
    .offset_factor = ${-(2**(W_data_size-1))},
    .offset_mode = weightOffsetModeLayerWise
  };

  nnx_feature_t nnx_input = {
    .data = db[i_db_x].x,
    .height = ${x_tile_size_h},
    .width = ${x_tile_size_w},
    .depth = ${x_tile_size_nif},
    .bitwidth = featureBitwidth8Bit
  };

  nnx_feature_t nnx_output = {
    .data = db[i_db_y].y,
    %if stride > 1:
    .height = y_tile_size_h_eff,
    .width = y_tile_size_w_eff,
    %else:
      .height = ${y_tile_size_h},
      .width = ${y_tile_size_w},
      %endif
    .depth = ${y_tile_size_nof},
    .bitwidth = featureBitwidth8Bit
  };

  const nnx_norm_t norm = {
    .mode  = normMode32Bit,
    .flag_bias  = FLAG_USED,
    .flag_shift = FLAG_UNUSED
  };

  const nnx_quant_t quant = {
    .shift_amount = out_shift,
    .mode = quantMode8Bit,
    .function = ${'quantFunctionRelu' if use_relu else 'quantFunctionIdentity'},
    .flag_rounding = FLAG_UNUSED
  };

  /////////////////
  // Total tiles //
  /////////////////

% if flag_DW == 0:
  const int total_tiles = ${tile_dim_nof} /*tile_dim_nof*/ * ${tile_dim_nif} /*tile_dim_nif*/ * ${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/;
% else:
  const int total_tiles = ${tile_dim_nof} /*tile_dim_nof*/ * ${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/;
% endif

  ///////////////////
  // NNX task init //
  ///////////////////

  for (int i = 0; i < MIN(NNX_TASK_COUNT, total_tiles); i++) {
    nnx_task_init(&nnx_tasks[i]);
    //nnx_pad_input(&(nnx_tasks[i].cfg),0,1,0,0,0);
    nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}(&(nnx_tasks[i].cfg), nnx_weights, nnx_input, nnx_output);
    nnx_norm_quant(&(nnx_tasks[i].cfg), norm, quant);
    % if use_wmem:
    BIT_SET(nnx_tasks[i].cfg.conf0, NEUREKA_FLAG_USE_WMEM);
    % endif
  }


//  /$$$$$$$$ /$$$$$$ /$$       /$$$$$$$$       /$$        /$$$$$$   /$$$$$$  /$$$$$$$ 
// |__  $$__/|_  $$_/| $$      | $$_____/      | $$       /$$__  $$ /$$__  $$| $$__  $$
//    | $$     | $$  | $$      | $$            | $$      | $$  \ $$| $$  \ $$| $$  \ $$
//    | $$     | $$  | $$      | $$$$$         | $$      | $$  | $$| $$  | $$| $$$$$$$/
//    | $$     | $$  | $$      | $$__/         | $$      | $$  | $$| $$  | $$| $$____/ 
//    | $$     | $$  | $$      | $$            | $$      | $$  | $$| $$  | $$| $$      
//    | $$    /$$$$$$| $$$$$$$$| $$$$$$$$      | $$$$$$$$|  $$$$$$/|  $$$$$$/| $$      
//    |__/   |______/|________/|________/      |________/ \______/  \______/ |__/      

  for (int i_tile = 0; i_tile < total_tiles; i_tile++)
  {
    //NEUREKA_CG_ENABLE();
      
//   /$$$$$$   /$$$$$$  /$$   /$$ /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$ /$$$$$$$  /$$$$$$$$
//  /$$__  $$ /$$__  $$| $$$ | $$| $$_____/|_  $$_/ /$$__  $$| $$  | $$| $$__  $$| $$_____/
// | $$  \__/| $$  \ $$| $$$$| $$| $$        | $$  | $$  \__/| $$  | $$| $$  \ $$| $$      
// | $$      | $$  | $$| $$ $$ $$| $$$$$     | $$  | $$ /$$$$| $$  | $$| $$$$$$$/| $$$$$   
// | $$      | $$  | $$| $$  $$$$| $$__/     | $$  | $$|_  $$| $$  | $$| $$__  $$| $$__/   
// | $$    $$| $$  | $$| $$\  $$$| $$        | $$  | $$  \ $$| $$  | $$| $$  \ $$| $$      
// |  $$$$$$/|  $$$$$$/| $$ \  $$| $$       /$$$$$$|  $$$$$$/|  $$$$$$/| $$  | $$| $$$$$$$$
//  \______/  \______/ |__/  \__/|__/      |______/ \______/  \______/ |__/  |__/|________/

    const int x_tile_ptr     = db[i_db_x].x;

    % if not use_wmem:
    const int w_tile_ptr     = db[i_db_w].w;
    % else:
      const int w_tile_ptr     = ${func_name}_weights;
    % endif

% if FLAG_BATCHNORM == 1:
    const int scale_tile_ptr = db[i_db_w].scale;
    const int bias_tile_ptr  = db[i_db_w].bias;
% endif
    const int y_tile_ptr     = db[i_db_y].y;

    ///////////////////////
    // DMA configuration //
    ///////////////////////

    static uint8_t p_t, p_l, p_b, p_r = 0;
    static uint32_t x_offset = 0;
    
    if(i_h==0){
      p_t = ${padding_top};
    } else {
      p_t = 0;
    }
    if(i_w==0){
      p_l = ${padding_left};
    } else{
      p_l = 0;
    }
    if(i_h+1==(${tile_dim_h})){
      p_b = ${padding_bottom};
    }else{
      p_b = 0;
    }
    if(i_w+1==(${tile_dim_w})){
      p_r = ${padding_right};
    } else{
      p_r = 0;
    }
    
    if (is_load_x) {
      x_tile_size_h = (i_h + 1 == ${tile_dim_h}) ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_w = (i_w + 1 == ${tile_dim_w}) ? ${x_tile_size_w_last} : ${x_tile_size_w};
      x_length_nif_byte = (i_nif + 1 == ${tile_dim_nif}) ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};

      
      pad_offset_h = i_h > 0 ? ${padding_top} : 0;
      //pad_offset_w = i_w > 0 ? ${padding_left} : 0;
      pad_offset_w = i_w > 0 ? ${padding_left} : 0;
      
      DMA_copy_x.ext = dory_get_tile_3d(l2_x, i_h ,i_w, i_nif, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g}, ${conv_overlap1}, ${conv_overlap2}, 0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      //DMA_copy_x.loc = x_tile_ptr + (p_t*x_tile_size_w*${x_tile_size_nif}) + p_l*${x_tile_size_nif};
      x_offset = (p_t*(x_tile_size_w - (p_l * (${tile_dim_w} > 1)))*x_length_nif_byte + p_l*x_length_nif_byte);
      
      DMA_copy_x.loc = x_tile_ptr + x_offset;
      
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w - (p_l * (${tile_dim_w} > 1));
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
    }

    if (is_load_w) {
      W_tile_size_nof = (i_nof + 1 == ${tile_dim_nof}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      W_tile_size_nif = (i_nif + 1 == ${tile_dim_nif}) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};

      W_tile_ko_len = (i_nof + 1 == ${tile_dim_nof}) ? ${l1_W_tile_ko_len_last} : ${l1_W_tile_ko_len};

      %if not use_wmem:
      DMA_copy_W.ext = l2_W + ${l1_W_tile_ko_len * l1_W_tile_ki_size} * i_nof;
      DMA_copy_W.loc = w_tile_ptr;
      DMA_copy_W.length_1d_copy = W_tile_ko_len * ${l1_W_tile_ki_size};
      % endif

% if FLAG_BATCHNORM == 1:
      DMA_copy_k.ext = l2_scale + ${k_tile_size_byte_transfer} * i_nof;
      DMA_copy_k.loc = scale_tile_ptr;
      DMA_copy_k.length_1d_copy = W_tile_size_nof * ${int(act_dim_bit/8)};

      DMA_copy_lambda.ext = l2_bias + ${lambda_tile_size_byte_transfer} * i_nof;
      DMA_copy_lambda.loc = bias_tile_ptr;
      DMA_copy_lambda.length_1d_copy = W_tile_size_nof * ${int(act_dim_bit/8)};
% endif
    }
    
    y_tile_size_h = (i_h + 1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w = (i_w + 1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};

    %if stride > 1:
    y_tile_size_h_eff = y_tile_size_h * ${stride} - ((x_tile_size_h + p_b + (${tile_dim_h} == 1)*p_t) % ${stride});
    y_tile_size_w_eff = y_tile_size_w * ${stride} - ((x_tile_size_w + p_r + (${tile_dim_w} == 1)*p_l) % ${stride});
    % else:
	y_tile_size_h_eff = y_tile_size_h;
    y_tile_size_w_eff = y_tile_size_w;
    %endif

       %if stride > 1:
       wEffY[DMA_Y_INDEX(i_tile)] = y_tile_size_w_eff;
       %endif
	  
    y_length_nof_byte = (i_nof + 1 == ${tile_dim_nof}) ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
    
    DMA_copy_y[DMA_Y_INDEX(i_tile)].ext = dory_get_tile_3d(l2_y, i_h, i_w, i_nof, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    DMA_copy_y[DMA_Y_INDEX(i_tile)].loc = y_tile_ptr;
    DMA_copy_y[DMA_Y_INDEX(i_tile)].number_of_2d_copies = y_tile_size_h;
    DMA_copy_y[DMA_Y_INDEX(i_tile)].number_of_1d_copies = y_tile_size_w;
    DMA_copy_y[DMA_Y_INDEX(i_tile)].length_1d_copy = y_length_nof_byte;

    ////////////////////////
    // NEUREKA configuration //
    ////////////////////////
    int is_border_tile = 0
  % if tile_dim_nif != 1:
      || i_nif + 1 == ${tile_dim_nif}
  % endif
  % if tile_dim_h != 1:
      || i_h + 1 == ${tile_dim_h}
  % endif
  % if tile_dim_w != 1:
      || i_w + 1 == ${tile_dim_w}
  % endif
  % if tile_dim_nof != 1:
      || i_nof + 1 == ${tile_dim_nof}
  % endif
    ;

    nnx_task_to_offload = is_border_tile ? &nnx_tasks[NNX_TASK_REMAINDER] : &nnx_tasks[NNX_TASK_BODY];
    // Scheremo: Add Padding support
    nnx_input.height = x_tile_size_h;
    nnx_input.width = x_tile_size_w;
    nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}(&(nnx_task_to_offload->cfg), nnx_weights, nnx_input, nnx_output);
    nnx_pad_input(&((*nnx_task_to_offload).cfg), p_t, p_r, p_b, p_l, 0);
    nnx_task_to_offload->cfg.conf0 = nnx_task_to_offload->cfg.conf0 | ((uint32_t)(${RELU}<<23));
    % if signed:
    nnx_task_to_offload->cfg.conf0 = nnx_task_to_offload->cfg.conf0 | ((uint32_t)(1<<26));
    % endif
    % if stride > 1:
    nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}_update_dims(&(nnx_task_to_offload->cfg), y_tile_size_h_eff, y_tile_size_w_eff, W_tile_size_nof, W_tile_size_nif);
    % else:
    nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}_update_dims(&(nnx_task_to_offload->cfg), y_tile_size_h, y_tile_size_w, W_tile_size_nof, W_tile_size_nif);
    %endif
    
    nnx_task_to_offload->infeat_ptr = x_tile_ptr;
    nnx_task_to_offload->weights_ptr = w_tile_ptr - 0x10400000;
% if FLAG_BATCHNORM == 1:
    nnx_task_to_offload->scale_ptr = scale_tile_ptr;
    nnx_task_to_offload->scale_bias_ptr = bias_tile_ptr;
% endif
    nnx_task_to_offload->outfeat_ptr = y_tile_ptr;



//  /$$        /$$$$$$   /$$$$$$  /$$$$$$$ 
// | $$       /$$__  $$ /$$__  $$| $$__  $$
// | $$      | $$  \ $$| $$  \ $$| $$  \ $$
// | $$      | $$  | $$| $$$$$$$$| $$  | $$
// | $$      | $$  | $$| $$__  $$| $$  | $$
// | $$      | $$  | $$| $$  | $$| $$  | $$
// | $$$$$$$$|  $$$$$$/| $$  | $$| $$$$$$$/
// |________/ \______/ |__/  |__/|_______/ 
                                        
    // Acquire implicitly acts as a barrier that waits for the
    // accelerator to not be full i.e. have less than NNX_CONTEXT_SIZE
    // jobs commited.
    // This barrier is required before dma_memcpy so that we don't
    // overwrite the data being used by the accelerator.
    
    dma_copy_y_job_ids[DMA_Y_INDEX(i_tile)] = nnx_acquire();
    
    if (is_load_x) {
      dory_dma_memcpy_async(&DMA_copy_x);
      if (STARTED == 0){
	startCycles = pi_perf_cl_read(PI_PERF_CYCLES);
	STARTED=1;
      }
    }    if (is_load_w) {
      % if not use_wmem:
      dory_dma_memcpy_async(&DMA_copy_W);
      % endif
% if FLAG_BATCHNORM == 1:
      dory_dma_memcpy_async(&DMA_copy_k);
      dory_dma_memcpy_async(&DMA_copy_lambda);
% endif
    }


//   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$$  /$$$$$$$$
//  /$$__  $$|__  $$__//$$__  $$| $$__  $$| $$_____/
// | $$  \__/   | $$  | $$  \ $$| $$  \ $$| $$      
// |  $$$$$$    | $$  | $$  | $$| $$$$$$$/| $$$$$   
//  \____  $$   | $$  | $$  | $$| $$__  $$| $$__/   
//  /$$  \ $$   | $$  | $$  | $$| $$  \ $$| $$      
// |  $$$$$$/   | $$  |  $$$$$$/| $$  | $$| $$$$$$$$
//  \______/    |__/   \______/ |__/  |__/|________/

    // If the accelerator is running a job with an id greater then
    // the id of the tile we have to store, it means it has processed
    // the tile and its output can be stored to l2 memory.
    const int is_store = nnx_job_id() > dma_copy_y_job_ids[DMA_Y_INDEX(i_store_y)];

    if (is_store) {
      %if stride > 1:
      // SCHEREMO: Contract outputs w/ stride factor
      // nnx_wait_on_id(dma_copy_y_job_ids[DMA_Y_INDEX(i_store_y)]);
      contract_strided_output(i_store_y, DMA_copy_y, wEffY, dory_dma_channel);
      %endif
      
      dory_dma_memcpy_async(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
    }


//  /$$$$$$$$ /$$   /$$ /$$$$$$$$  /$$$$$$ 
// | $$_____/| $$  / $$| $$_____/ /$$__  $$
// | $$      |  $$/ $$/| $$      | $$  \__/
// | $$$$$    \  $$$$/ | $$$$$   | $$      
// | $$__/     >$$  $$ | $$__/   | $$      
// | $$       /$$/\  $$| $$      | $$    $$
// | $$$$$$$$| $$  \ $$| $$$$$$$$|  $$$$$$/
// |________/|__/  |__/|________/ \______/ 
    
    nnx_offload(nnx_task_to_offload);
    
    // Wait for data to arrive
    if (is_load_x) {
      dory_dma_barrier(&DMA_copy_x);
    }
   
    if (is_load_w) {
      % if not use_wmem:
      dory_dma_barrier(&DMA_copy_W);
      % endif
% if FLAG_BATCHNORM == 1:
      dory_dma_barrier(&DMA_copy_k);
      dory_dma_barrier(&DMA_copy_lambda);
% endif
    }
    // This checks if we are about to start a job that is writting
    // to the buffer that we are storing at the moment.
    if (i_tile == i_store_y + 2) {
      dory_dma_barrier(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
    }

    //print_task(*nnx_task_to_offload);
#ifdef RUNTIMEMEASUREMENT
    cycles = pi_perf_read(PI_PERF_CYCLES);
    nnx_run_blocking();
    tile_cycles = pi_perf_read(PI_PERF_CYCLES) - cycles;
    printf("\r\nRUNTIMEMEASUREMENT ${func_name} Tile %u: %u \r\n", i_tile, tile_cycles);
    tot_cycles += tile_cycles;
#endif
#ifndef RUNTIMEMEASUREMENT
    nnx_run_async();
#endif
    nnx_cfg_t cfg = nnx_task_to_offload->cfg;
    /* printf("in feat d0:\t\t %08x\r\n", cfg.input_stride.d0); */
    /* printf("in feat d1:\t\t %08x\r\n", cfg.input_stride.d1); */
    /* printf("in feat d2:\t\t %08x\r\n", cfg.input_stride.d2); */

    /* printf("out feat d0:\t\t %08x\r\n", cfg.output_stride.d0); */
    /* printf("out feat d1:\t\t %08x\r\n", cfg.output_stride.d1); */
    /* printf("out feat d2:\t\t %08x\r\n", cfg.output_stride.d2); */

    /* printf("weights d0:\t\t %08x\r\n", cfg.weights_stride.d0); */
    /* printf("weights d1:\t\t %08x\r\n", cfg.weights_stride.d1); */
    /* printf("weights d2:\t\t %08x\r\n", cfg.weights_stride.d2); */

    /* printf("subtile KoKi:\t\t %08x\r\n", cfg.subtile.remainder.KoKi); */
    /* printf("subtile HoWo:\t\t %08x\r\n", cfg.subtile.remainder.HoWo); */
    /* printf("subtile HiWi:\t\t %08x\r\n", cfg.subtile.remainder.HiWi); */

    /* printf("subtile KoKi:\t\t %08x\r\n", cfg.subtile.number.KoKi); */
    /* printf("subtile HoWo:\t\t %08x\r\n", cfg.subtile.number.HoWo); */

    /* printf("Padding:\t\t %08x\r\n", cfg.padding); */
    /* printf("weight_offset:\t\t %08x\r\n", cfg.weight_offset_factor); */
    /* printf("filter_mask:\t\t %08x\r\n", cfg.filter_mask); */
    /* printf("conf0:\t\t\t %08x\r\n", cfg.conf0); */
    
    //nnx_run_blocking();


//  /$$   /$$ /$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$$$       /$$$$$$ /$$   /$$ /$$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$$$  /$$$$$$ 
// | $$  | $$| $$__  $$| $$__  $$ /$$__  $$|__  $$__/| $$_____/      |_  $$_/| $$$ | $$| $$__  $$|_  $$_/ /$$__  $$| $$_____/ /$$__  $$
// | $$  | $$| $$  \ $$| $$  \ $$| $$  \ $$   | $$   | $$              | $$  | $$$$| $$| $$  \ $$  | $$  | $$  \__/| $$      | $$  \__/
// | $$  | $$| $$$$$$$/| $$  | $$| $$$$$$$$   | $$   | $$$$$           | $$  | $$ $$ $$| $$  | $$  | $$  | $$      | $$$$$   |  $$$$$$ 
// | $$  | $$| $$____/ | $$  | $$| $$__  $$   | $$   | $$__/           | $$  | $$  $$$$| $$  | $$  | $$  | $$      | $$__/    \____  $$
// | $$  | $$| $$      | $$  | $$| $$  | $$   | $$   | $$              | $$  | $$\  $$$| $$  | $$  | $$  | $$    $$| $$       /$$  \ $$
// |  $$$$$$/| $$      | $$$$$$$/| $$  | $$   | $$   | $$$$$$$$       /$$$$$$| $$ \  $$| $$$$$$$/ /$$$$$$|  $$$$$$/| $$$$$$$$|  $$$$$$/
//  \______/ |__/      |_______/ |__/  |__/   |__/   |________/      |______/|__/  \__/|_______/ |______/ \______/ |________/ \______/ 

    /////////////////////////
    // Update tile indices //
    /////////////////////////

% if tile_dim_nif != 1:
    const int i_nif_prev = i_nif;
% endif
% if tile_dim_w != 1:
    const int i_w_prev = i_w;
% endif
% if tile_dim_h != 1:
    const int i_h_prev = i_h;
% endif
% if tile_dim_nof != 1:
    const int i_nof_prev = i_nof;
% endif

% if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    i_nif += 1;
    if(i_nif==${tile_dim_nif}) {
      i_nif = 0;
% endif
% if tile_dim_w != 1:
      i_w += 1;
      if(i_w==${tile_dim_w}) {
        i_w = 0;
% endif
% if tile_dim_h != 1:
        i_h += 1;
        if(i_h==${tile_dim_h}) {
          i_h = 0;
% endif
% if flag_DW == 1:
          i_nif += 1;
% endif
% if tile_dim_nof != 1:
          i_nof += 1;
% endif
% if tile_dim_h != 1:
        }
% endif
% if tile_dim_w != 1:
      }
% endif
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif

    ///////////////////////
    // Update load flags //
    ///////////////////////

    is_load_w = 0
  % if tile_dim_nif != 1:
      || i_nif_prev != i_nif
  % endif
  % if tile_dim_nof != 1:
      || i_nof_prev != i_nof
  % endif
    ;

    is_load_x = 0
  % if tile_dim_nif != 1:
      || i_nif_prev != i_nif
  % endif
  % if tile_dim_h != 1:
      || i_h_prev != i_h
  % endif
  % if tile_dim_w != 1:
      || i_w_prev != i_w
  % endif
    ;

    ///////////////////////////
    // Update store iterator //
    ///////////////////////////
    if (is_store) {
      i_store_y += 1;
    }

    ///////////////////////////////////
    // Update double buffer pointers //
    ///////////////////////////////////

    if (is_load_x) i_db_x = !i_db_x;
    if (is_load_w) i_db_w = !i_db_w;
    i_db_y = !i_db_y;
    
    /* printf("Input: \r\n"); */
    /* for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)x_tile_ptr)[i]); */
    /* } */
    /* printf("\r\n"); */
    /* printf("Input @ linebreak: \r\n"); */
    /* for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)x_tile_ptr)[((x_tile_size_w)*x_length_nif_byte) + i]); */
    /* } */
    /* printf("\r\n"); */
    /* printf("Input @ linebreak 2: \r\n"); */
    /* for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)x_tile_ptr)[((2*x_tile_size_w)*x_length_nif_byte) + i]); */
    /* } */
    /* printf("\r\n"); */
    /* nnx_wait_empty(); */

    /* printf("Weight:\r\n"); */
    /*   for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)w_tile_ptr)[i]); */
    /* } */
    /* printf("\r\n"); */

    /* printf("Scale:\r\n"); */
    /*   for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)scale_tile_ptr)[i]); */
    /* } */
    /* printf("\r\n"); */

    /* printf("Bias:\r\n"); */
    /*   for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)bias_tile_ptr)[i]); */
    /* } */
    /* printf("\r\n"); */
    
    /* printf("Output1: \r\n"); */
    /* for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)y_tile_ptr)[i]); */
    /* } */
    /* printf("...\r\n"); */
    /* for (int i=0;i<30;i++){ */
    /*   printf("%u, ", ((uint8_t*)y_tile_ptr)[y_tile_size_w*y_tile_size_h-30 + i]); */
    /* } */
    /* printf("\r\n"); */
  }


//   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$$  /$$$$$$$$       /$$$$$$$  /$$$$$$$$ /$$      /$$
//  /$$__  $$|__  $$__//$$__  $$| $$__  $$| $$_____/      | $$__  $$| $$_____/| $$$    /$$$
// | $$  \__/   | $$  | $$  \ $$| $$  \ $$| $$            | $$  \ $$| $$      | $$$$  /$$$$
// |  $$$$$$    | $$  | $$  | $$| $$$$$$$/| $$$$$         | $$$$$$$/| $$$$$   | $$ $$/$$ $$
//  \____  $$   | $$  | $$  | $$| $$__  $$| $$__/         | $$__  $$| $$__/   | $$  $$$| $$
//  /$$  \ $$   | $$  | $$  | $$| $$  \ $$| $$            | $$  \ $$| $$      | $$\  $ | $$
// |  $$$$$$/   | $$  |  $$$$$$/| $$  | $$| $$$$$$$$      | $$  | $$| $$$$$$$$| $$ \/  | $$
//  \______/    |__/   \______/ |__/  |__/|________/      |__/  |__/|________/|__/     |__/

  for (; i_store_y < total_tiles; i_store_y++) {
    if (i_store_y < total_tiles - 1) {
      nnx_wait_on_id(dma_copy_y_job_ids[DMA_Y_INDEX(i_store_y)]);
    } else {
      nnx_wait_empty();
    }
      %if stride > 1:
    contract_strided_output(i_store_y, DMA_copy_y, wEffY, dory_dma_channel);
      %endif
    dory_dma_memcpy_async(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
  }

  endCycles = pi_perf_cl_read(PI_PERF_CYCLES);
  ${prefix}NEUREKA_CYCLES += endCycles - startCycles;

#ifdef RUNTIMEMEASUREMENT
  printf("\r\nRUNTIMEMEASUREMENT ${func_name}: %u \r\n", tot_cycles);
#endif
  
% if not TEST:
  // wait for final write
  dory_dma_barrier(&DMA_copy_y[DMA_Y_INDEX(total_tiles-1)]);
  dory_dma_free(dory_dma_channel);
  
% endif
  // clear NNX for cleanup
    //NEUREKA_CG_DISABLE();
    //nnx_soft_clear();
}

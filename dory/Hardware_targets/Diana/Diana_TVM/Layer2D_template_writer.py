#
# template.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from mako.template import Template
import re
from collections import OrderedDict
import numpy as np
import sys
import os
import re

def print_template_layer(tk,node, layer_type, tmpl_dir, out_dir, double_buffering = 2):
    ks =      node.kernel_shape
    inp_dim = node.tiling_dimensions["L2"]["input_dimensions"][1:]
    out_dim = node.tiling_dimensions["L2"]["output_dimensions"][1:]
    in_ch =   node.tiling_dimensions["L2"]["input_dimensions"][0]
    s =       node.strides
    g =       node.group
    p =       node.pads
    conv_overlap_h = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    padding_top = p[0]; padding_left = p[1]; padding_bottom = p[2]; padding_right = p[3];
    name_layer = node.name + '.h'
    conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)
    if (re.search('.0',name_layer)):
        try:
            int(re.search('.0',name_layer).group())
            tk['first_layer'] = 0
        except ValueError:
            tk['first_layer'] = 1
    else:
        tk['first_layer'] = 0
    tk['sdk'] = node.HW_description["software development kit"]["name"]
    tk['number_of_clusters'] = node.HW_description["number_of_clusters"] if "number_of_clusters" in node.HW_description.keys() else 1
    tk['optional_type'] = layer_type
    tk['func_name'] = node.name
    tk['flag_DW'] = 1 if node.group > 1 else 0
    tk['optional'] = node.op_type
    tk['FLAG_BATCHNORM'] = 1 if 'k' in node.constant_names else 0
    tk['has_bias'] = int(len([1 for name in node.constant_names if "bias" in name])>0)
    tk['FLAG_RELU'] = 1 if 'outshift' in node.constant_names else 0
    tk['type'] = "char" if node.input_activation_type in ["int", "uint"] else "float"
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding_top'] = padding_top
    tk['padding_bottom'] = padding_bottom
    tk['padding_left'] = padding_left
    tk['padding_right'] = padding_right
    tk['stride'] = s[0]

    ################## NEED A REWRITING IN THIS TEMPLATE PART ######################
    #### VARIABLE CREATION FOR COMPATIBILITY WITH THE SECTION AFTER ################
    if tk['flag_DW'] == 0:
        tk['g'] = 1
        tk['nif'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
    else:
        tk['g'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
        tk['nif'] = 1
    n_in       = node.tiling_dimensions["L2"]["input_dimensions"][0]
    h_in       = node.tiling_dimensions["L2"]["input_dimensions"][1]
    w_in       = node.tiling_dimensions["L2"]["input_dimensions"][2]
    tile_n_in  = node.tiling_dimensions["L1"]["input_dimensions"][0]
    tile_h_in  = node.tiling_dimensions["L1"]["input_dimensions"][1]
    tile_w_in  = node.tiling_dimensions["L1"]["input_dimensions"][2]

    if "Addition" not in node.name and "Pool" not in node.name:
        n_out  = node.tiling_dimensions["L2"]["weights_dimensions"][0]
    else:
        n_out  = node.tiling_dimensions["L2"]["output_dimensions"][0]
    h_out      = node.tiling_dimensions["L2"]["output_dimensions"][1]
    w_out      = node.tiling_dimensions["L2"]["output_dimensions"][2]
    tile_n_out = node.tiling_dimensions["L1"]["output_dimensions"][0]
    tile_h_out = node.tiling_dimensions["L1"]["output_dimensions"][1]
    tile_w_out = node.tiling_dimensions["L1"]["output_dimensions"][2]
 
    fs1        = node.kernel_shape[0]
    fs2        = node.kernel_shape[1]

    ds_x       = node.input_activation_bits
    ds_y       = node.output_activation_bits
    ds_act     = node.constant_bits
    ds_W       = node.weight_bits
    ds_bias    = node.bias_bits

    dt_x       = node.input_activation_type
    dt_y       = node.output_activation_type
    dt_act     = node.constant_type
    dt_W       = node.weight_type

    if "Addition" in node.name:
        ds_x2  = node.input_activation_bits
        dt_x2  = node.input_activation_type
        tk["data_type_x2"] = dt_x2
        tk['x_data_size_byte2'] = ds_x2
        tk["inmul1"] = node.inmul1["value"]
        tk["inadd1"] = node.inadd1["value"]
        tk["inshift1"] = node.inshift1["value"]
        tk["inmul2"] = node.inmul2["value"]
        tk["inadd2"] = node.inadd2["value"]
        tk["inshift2"] = node.inshift2["value"]
        tk["outmul"] = node.outmul["value"]
        tk["outadd"] = node.outadd["value"]
        tk["outshift"] = node.outshift["value"]
    tk['out_shift'] = node.outshift["value"] if 'outshift' in node.constant_names else 0

    DW = tk['flag_DW']
    has_bias = tk['has_bias']
    number_of_clusters = tk['number_of_clusters']

    tk["data_type_x"] = dt_x
    tk["data_type_y"] = dt_y
    tk["data_type_activations"] = dt_act
    tk["data_type_weights"] = dt_W
    ################################################################################

    tk['nof'] = n_out
    if node.HW_description['memory']['levels'] > 2:
        tk['factor'] = node.tiling_dimensions["L3"]["output_dimensions"][0] / n_out
    else:
        tk['factor'] = 1
    # x parameters
    tk['double_buffering'] = double_buffering
    tk['x_h'] = h_in
    tk['x_w'] = w_in
    tk['x_data_size_byte'] = node.input_activation_bits
    tk['x_tile_size_nif'] = tile_n_in
    tk['x_tile_size_h'] = tile_h_in
    tk['x_tile_size_w'] = tile_w_in
    tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    tk['x_tile_size_nif_byte'] = int(math.ceil(tile_n_in * ds_x / 8.0))
    tk['x_stride_w_byte'] = int(math.ceil(w_in * n_in * ds_x / 8.0))
    tk['x_stride_c_byte'] = int(math.ceil(n_in * ds_x / 8.0))
    # y parameters
    tk['y_h'] = h_out
    tk['y_w'] = w_out
    tk['y_data_size_byte'] = ds_y
    tk['act_dim_bit'] = ds_act
    tk['y_tile_size_nof'] = tile_n_out if (n_out > tile_n_out) else n_out
    tk['y_tile_size_h'] = tile_h_out if (h_out > tile_h_out) > 0 else h_out
    tk['y_tile_size_w'] = tile_w_out if (w_out > tile_w_out) > 0 else w_out
    tk['y_tile_size_byte'] = int(math.ceil(tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] * ds_y / 8.0))
    tk['y_stride_w_byte'] = int(math.ceil(w_out * n_out * tk['factor'] * ds_y / 8.0))
    tk['y_stride_c_byte'] = int(math.ceil(n_out * tk['factor'] * ds_y / 8.0))
    tk['y_tile_size_nof_byte'] = int(math.ceil(tile_n_out * ds_y / 8.0))

    tk['tile_dim_h'] = max(int(math.ceil(float(h_out) / float(tk['y_tile_size_h']))), 1)
    tk['tile_dim_w'] = max(int(math.ceil(float(w_out) / float(tk['y_tile_size_w']))), 1)
    tk['tile_dim_nof'] = max(int(math.ceil(float(n_out) / float(tk['y_tile_size_nof']))), 1)
    tk['tile_dim_nif'] = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
    tk['tile_n_in_last'] = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
    # W parameters
    tk['fs1'] = fs1
    tk['fs2'] = fs2
    tk['W_data_size_byte'] = ds_W
    tk['W_tile_size_nof'] = tile_n_out 
    if tk['has_bias'] == 1:
        tk['b_size_byte'] = int(math.ceil(n_out * ds_bias / 8.0))
    else:
        tk['b_size_byte'] = 0

    if DW == 0:
        tk['W_tile_size_nif'] = tile_n_in * tk['tile_dim_nif']
        tk['W_tile_size_nif_last'] = tk['tile_n_in_last'] * tk['tile_dim_nif']
    else:
        tk['W_tile_size_nif'] = 1
        tk['W_tile_size_nif_last'] = 1
    if "Addition" not in node.name and "Pool" not in node.name:
        tk['W_tile_size_byte'] = int(math.ceil(tile_n_out * tk['W_tile_size_nif'] * fs1 * fs2 * ds_W / 8.0))
        if DW == 0:
            tk['W_stride_nof_byte'] = int(math.ceil(tk['nif'] * fs1 * fs2 * ds_W / 8.0))
        else:
            tk['W_stride_nof_byte'] = int(math.ceil(tk['nif'] * fs1 * fs2 * ds_W / 8.0))        
        tk['W_stride_hw_byte'] = int(math.ceil(tk['nif'] * ds_W / 8.0))
        tk['W_tile_nif_byte'] = int(math.ceil(tk['W_tile_size_nif'] * ds_W / 8.0))
        tk['W_tile_nif_byte_last'] = int(math.ceil(tk['W_tile_size_nif_last'] * ds_W / 8.0))
    # l2 parameters
    if tk['FLAG_BATCHNORM'] == 1:
        tk['l2_off_k'] = int(
            math.ceil(tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W / 8.0 + tk['b_size_byte']))
        tk['l2_off_lambda'] = int(
            math.ceil((tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W + tk['nof'] * ds_act) / 8.0 + tk['b_size_byte']))
    if has_bias == 1:
        tk['l2_off_bias'] = int(math.ceil(tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W / 8.0 ))
    if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in:
        x_buffer_size = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    else:
        x_buffer_size = tk['double_buffering'] * int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
        if x_buffer_size % 16 != 0:
            x_buffer_size = x_buffer_size
    if (n_in == (tile_n_in * number_of_clusters) and w_in == tile_w_in and h_in == tile_h_in and n_out == (tile_n_out * number_of_clusters) and n_in > number_of_clusters) \
    or (n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == (tile_n_out * number_of_clusters)):
        y_buffer_size = int(math.ceil(ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] / 8.0))
        if "Addition" not in node.name and "Pool" not in node.name:
            if DW == 0:
                W_buffer_size = int(math.ceil(ds_W * tk['y_tile_size_nof']  * tk['W_tile_size_nif'] * fs1 * fs2 / 8.0))
            else:
                W_buffer_size = int(math.ceil(ds_W * tk['y_tile_size_nof']  * 1 * fs1 * fs2 / 8.0))
        else:
            W_buffer_size = 0
    else:
        y_buffer_size = tk['double_buffering'] * int(math.ceil(ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] / 8.0))
        if "Addition" not in node.name and "Pool" not in node.name:
            if DW == 0:
                W_buffer_size = tk['double_buffering'] * int(math.ceil(ds_W * tk['y_tile_size_nof'] * tk['W_tile_size_nif'] * fs1 * fs2 / 8.0))
            else:
                W_buffer_size = tk['double_buffering'] * int(math.ceil(ds_W * tk['y_tile_size_nof'] * 1 * fs1 * fs2 / 8.0))
        else:
            W_buffer_size = 0
    if tk['FLAG_BATCHNORM'] == 1:
        k_buffer_size = int(n_out * ds_act / 8.0)
        lambd_buffer_size = int(n_out * ds_act / 8.0)
    else:
        k_buffer_size = 0
        lambd_buffer_size = 0

    tk['k_tile_size_byte'] = 0
    tk['lambda_tile_size_byte'] = 0
    tk['k_size_byte'] = 0
    tk['lambda_size_byte'] = 0
    if "Pool" not in node.name:
        if tk['FLAG_BATCHNORM'] == 1:
            tk['k_size_byte'] = k_buffer_size
            tk['lambda_size_byte'] = k_buffer_size
            tk['k_tile_size_byte_transfer'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            tk['lambda_tile_size_byte_transfer'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == tile_n_out:
                tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
                tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            else:
                tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0 * tk['double_buffering']))
                tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0 * tk['double_buffering']))
        if has_bias == 1:
            tk['bias_tile_size_byte'] = tile_n_out * int(ds_bias / 8.0)
            tk['b_size_byte'] = int(n_out)* int(ds_bias / 8.0)
        else:
            tk['bias_tile_size_byte'] = 0
            tk['b_size_byte'] = 0

    # l1 parameters
    tk['l1_x_offset'] = 0
    tk['l1_y_offset'] = x_buffer_size + 8
    if "Addition" in node.name:
        tk['l1_x2_offset'] = x_buffer_size + 8 + y_buffer_size + 8
    if "Addition" not in node.name and "Pool" not in node.name:
        tk['l1_W_offset'] = x_buffer_size + 8 + y_buffer_size + 8
        if tk['FLAG_BATCHNORM'] == 1:
            tk['l1_k_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8
            tk['l1_lambda_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + tk['k_tile_size_byte'] + 8
        if has_bias == 1:
            tk['l1_b_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + tk['k_tile_size_byte'] + 8 + tk['lambda_tile_size_byte']  + 8

    # W last
    if "Addition" not in node.name and "Pool" not in node.name:
        tk['W_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
        tk['W_tile_size_nif_last'] = tk['W_tile_size_nif']
        tk['W_tile_size_nif_byte_last'] = int(math.ceil(tk['W_tile_size_nif_last'] * ds_W / 8.0))
    
    # y last
    tk['y_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
    tk['y_tile_size_h_last'] = h_out % tile_h_out if (h_out % tile_h_out) > 0 else tile_h_out
    tk['y_tile_size_w_last'] = w_out % tile_w_out if (w_out % tile_w_out) > 0 else tile_w_out
    tk['y_length_nof_byte_last'] = int(math.ceil(tk['y_tile_size_nof_last'] * ds_y / 8.0))

    # x last
    tk['x_tile_size_nif_last'] = n_in % tile_n_in if (n_in % tile_n_in) > 0 else tile_n_in
    tk['x_tile_size_nif_byte_last'] = int(math.ceil(tk['x_tile_size_nif_last'] * ds_x / 8.0))
    tk['x_tile_size_h_last'] = tk['y_tile_size_h_last'] * s[0] + ks[0] - s[0] - (padding_bottom - ((h_in + padding_bottom + padding_top) - (h_out* s[0] + ks[0] - s[0])))
    tk['x_tile_size_w_last'] = tk['y_tile_size_w_last'] * s[1] + ks[1] - s[1] - (padding_right - ((w_in + padding_left + padding_right) - (w_out* s[1] + ks[1] - s[1])))
    ## single tile execution
    if tk['x_tile_size_h_last'] > tk['x_tile_size_h']:
        tk['x_tile_size_h_last'] = tk['x_tile_size_h']
    if tk['x_tile_size_w_last'] > tk['x_tile_size_w']:
        tk['x_tile_size_w_last'] = tk['x_tile_size_w']

    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    if "Addition" not in node.name and "Pool" not in node.name:
        buffer_l1_all = W_buffer_size + x_buffer_size + y_buffer_size + tk['k_tile_size_byte'] + tk['lambda_tile_size_byte'] + 40 + tk['b_size_byte']
        tk['im2col_dim'] = (8 * (fs1 * (tile_h_in + padding_bottom + padding_top) + fs1)) * int( 8 / min(ds_x, ds_y, ds_W))
    elif "Addition" in node.name:
        buffer_l1_all = x_buffer_size * tk['double_buffering'] + y_buffer_size + tk['k_tile_size_byte'] + tk['lambda_tile_size_byte'] + 40 + tk['b_size_byte']
    elif "Pool" in node.name:
        buffer_l1_all = x_buffer_size + y_buffer_size + tk['k_tile_size_byte'] + tk['lambda_tile_size_byte'] + 40 + tk['b_size_byte']
    tk['buffer_l1_all'] = buffer_l1_all

    # only used for avg pool layers
    tk['out_add'] = node.outadd["value"] if 'outadd' in node.constant_names else 0
    tk['out_mul'] = node.outmul["value"] if 'outmul' in node.constant_names else 1

    if "Addition" not in node.name and "Pool" not in node.name:
        tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L2_c_conv_template.c"))
    elif "Pool" in node.name:
        tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L2_c_pooling_template.c"))
    elif "Addition" in node.name:
        tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L2_c_addition_template.c"))

    s_c = tmpl.render(verbose_log=l, **tk)
    return s_c


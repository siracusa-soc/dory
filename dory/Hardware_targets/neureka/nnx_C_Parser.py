# should work even without -*-
# -*- coding: utf-8 -*-
# !/bin/bash
# ONNX_to_DORY_generic.py
# Alessio Burrello <alessio.burrello@unibo.it>
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

# Libraries
import json
import os
import shutil
import numpy as np

# DORY modules
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
from dory.Utils.Templates_writer.TemplateWriter2D import TemplateWriter2D_L2
from .Util import div_and_ceil, rem


class nnx_C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, confdir, verbose_level, perf_layer, precision_library, app_directory, accelerator):
        self.nnxdir = os.path.dirname(__file__)
        with open(os.path.join(self.nnxdir, 'HW_description.json'), 'r') as f:
            hw_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = conf["BNRelu_bits"]
        self.config_file = conf
        self.acc = accelerator
        super().__init__(graph, os.path.join(confdir, os.path.dirname(conf["onnx_file"])),
                         hw_description, verbose_level, perf_layer, "Makefile", app_directory)

    @staticmethod
    def copy_backend_files(node, app_directory, nnxdir, acc_name):
        out_dir = app_directory
        out_inc_dir = os.path.join(out_dir, 'inc')
        out_src_dir = os.path.join(out_dir, 'src')
        backend_dir = os.path.join(nnxdir, 'Backend_Kernels', 'pulp-nnx')
        backend_inc_dir = os.path.join(backend_dir, 'include')
        backend_src_dir = os.path.join(backend_dir, 'src')

        def cp_files(src_dir, dest_dir, files):
            for file in files:
                shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

        def cp_dir_files(src_dir, dest_dir):
            for (dirpath, dirnames, filenames) in os.walk(src_dir):
                cp_files(dirpath, dest_dir, filenames)
                if acc_name in dirnames:
                    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dirpath, acc_name)):
                        cp_files(dirpath, dest_dir, filenames)
                        break
                break

        cp_dir_files(backend_inc_dir, out_inc_dir)
        cp_dir_files(backend_src_dir, out_src_dir)

    @staticmethod
    def map_layer_to_C_file(node, config_file, acc, tmpl_dir, out_dir, HW_description):
        tmpl_writer = TemplateWriter2D_L2(node)
        tmpl_writer = nnx_C_Parser.__nnx_vars(tmpl_writer, node, config_file, acc, HW_description)
        tmpl_writer.use_wmem = config_file['use_wmem']
        tmpl_files = ['layer_L2_h_template.h', 'layer_L2_c_conv_template.c']
        tmpl_files = [os.path.join(tmpl_dir, tmpl_file) for tmpl_file in tmpl_files]

        #import IPython; IPython.embed()
        tmpl_writer.write(tmpl_files, out_dir)

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.abspath(os.path.join(self.nnxdir, 'Templates', 'layer_templates'))
        out_dir = self.app_directory

        for node in self.HWgraph:
            nnx_C_Parser.copy_backend_files(node, self.app_directory, self.nnxdir, self.acc.name)
            nnx_C_Parser.map_layer_to_C_file(node, self.config_file, self.acc, tmpl_dir, out_dir, self.HW_description)

    @staticmethod
    def __mem_tmpl_vars(tmpl_writer, node, mem_level, config_file, acc, HW_description):
        mem_name = f'L{mem_level}'
        upper_mem_name = f'L{mem_level + 1}'
        no_w_tiling = mem_level == 1 and config_file['use_wmem'] and HW_description['memory']['wmem']
        def set_tmpl_var(name, val):
            prefix = f'{mem_name.lower()}_'

            def attr(var_name):
                return f'{prefix}{var_name}'

            setattr(tmpl_writer, attr(name), val)

        flag_depthwise = node.group > 1
        flag_batchnorm = 'k' in node.constant_names
        flag_bias = len([1 for name in node.constant_names if 'bias' in name]) > 0

        input_tile_shape = node.tiling_dimensions[mem_name]['input_dimensions']
        output_tile_shape = node.tiling_dimensions[mem_name]['output_dimensions']
        weights_tile_shape = node.tiling_dimensions[mem_name]['weights_dimensions']

        weights_tile_ko, weights_tile_ki = weights_tile_shape

        weights_tile_size = 0 if no_w_tiling else acc.weights_size(weights_tile_ko, weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)
        set_tmpl_var('W_size', weights_tile_size)

        input_el_size = div_and_ceil(node.input_activation_bits, 8)
        output_el_size = div_and_ceil(node.output_activation_bits, 8)
        activation_el_size = div_and_ceil(node.constant_bits, 8)

        tile_ko = weights_tile_ki if flag_depthwise else weights_tile_ko
        weights_tile_ko_len = 0 if no_w_tiling else acc.weights_ko_len(tile_ko, flag_depthwise)
        weights_tile_ki_size = 0 if no_w_tiling else acc.weights_ki_size(weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)

        def feature_len(shape):
            return shape[0] * shape[1] * shape[2]

        if upper_mem_name in node.tiling_dimensions.keys():
            input_shape = node.tiling_dimensions[upper_mem_name]['input_dimensions']
            output_shape = node.tiling_dimensions[upper_mem_name]['output_dimensions']
            weights_shape = node.tiling_dimensions[upper_mem_name]['weights_dimensions']

            n_buffers = {
                'x': 1 if input_shape == input_tile_shape else 2,
                'y': 1 if output_shape == output_tile_shape else 2,
                'W': 1 if weights_shape == weights_tile_shape else 2,
                'k': 1 if weights_shape == weights_tile_shape else 2,
                'lambda': 1 if weights_shape == weights_tile_shape else 2,
                'b': 1 if weights_shape == weights_tile_shape else 2,
            }
        else:
            n_buffers = {'x': 1, 'y': 1, 'W': 1, 'k': 1, 'lambda': 1, 'b': 1}

        effective_x_tile_size = [input_tile_shape[0]]
        for idx, dim in enumerate(input_tile_shape[1:]):
            effective_x_tile_size += [dim + node.pads[idx]]

        tile_sizes = {
            'x': feature_len(effective_x_tile_size) * input_el_size,
            'y': feature_len(output_tile_shape)*int(np.prod(node.strides)) * output_el_size,
            'W': weights_tile_ko_len * weights_tile_ki_size,
            'k': tile_ko * activation_el_size,
            'lambda': tile_ko * activation_el_size,
            'b': tile_ko * div_and_ceil(node.weight_bits, 8)
        }

        # SCHEREMO: Add padding to x buffer
        buffer_sizes = {key: tile_sizes[key] * n_buffers[key] for key in tile_sizes.keys()}

        data_arrays = ['x', 'y', 'W']

        if flag_batchnorm:
            data_arrays.append('k')
            data_arrays.append('lambda')

        if flag_bias:
            data_arrays.append('b')

        offset = 0

        for data_array in data_arrays:
            set_tmpl_var(f'{data_array}_offset', offset)
            set_tmpl_var(f'{data_array}_tile_size', tile_sizes[data_array])
            offset += buffer_sizes[data_array]

        if upper_mem_name in node.tiling_dimensions.keys():
            input_shape = node.tiling_dimensions[upper_mem_name]['input_dimensions']
            output_shape = node.tiling_dimensions[upper_mem_name]['output_dimensions']
            weights_shape = node.tiling_dimensions[upper_mem_name]['weights_dimensions']

            input_depth, input_height, input_width = input_shape
            output_depth, output_height, output_width = output_shape
            weights_ko, weights_ki = weights_shape

            ko = weights_ki if flag_depthwise else weights_ko
            weights_ko_len = acc.weights_ko_len(ko, flag_depthwise)

            set_tmpl_var('W_tile_ko_len', weights_tile_ko_len)
            set_tmpl_var('W_tile_ko_len_last', 0 if no_w_tiling else rem(weights_ko_len, weights_tile_ko_len))
            set_tmpl_var('W_tile_ki_size', weights_tile_ki_size)

            x_dma_stride_1d = input_depth * input_el_size
            x_dma_stride_2d = input_width * x_dma_stride_1d
            set_tmpl_var('x_dma_stride_1d', x_dma_stride_1d)
            set_tmpl_var('x_dma_stride_2d', x_dma_stride_2d)

            y_dma_stride_1d = output_depth * output_el_size
            y_dma_stride_2d = output_width * y_dma_stride_1d
            set_tmpl_var('y_dma_stride_1d', y_dma_stride_1d)
            set_tmpl_var('y_dma_stride_2d', y_dma_stride_2d)

    @staticmethod
    def __nnx_vars(tmpl_writer, node, config_file, acc, HW_description):
        nnx_C_Parser.__mem_tmpl_vars(tmpl_writer, node, 1, config_file, acc, HW_description)
        nnx_C_Parser.__mem_tmpl_vars(tmpl_writer, node, 2, config_file, acc, HW_description)
        return tmpl_writer

    def create_hex_weight(self, node):
        # if not hasattr(node, "offloadable") or not node.offloadable:
        super().create_hex_weight(node)
        if self.config_file["use_wmem"]:
            constants = [0, 0, 0, 0]
            for name in node.constant_names:
                if "weight" in name:
                    constants[0] = name
                elif "bias" in name:
                    constants[1] = name
                elif "k" == name:
                    constants[2] = name
                elif "l" == name:
                    constants[3] = name

            weights = bytearray()
            for const in constants[:1]:
                if const != 0:
                    weights += getattr(node, const)['value'].tobytes()

            if len(weights) % 4 != 0:
                weights += bytearray([0] * (4 - len(weights) % 4))

            weightstr = ''
            weightstr += f"#include \"{node.name}_weights.h\"\r\n"
            weightstr += f"#include \"pmsis.h\"\r\n"
            weightstr += '__attribute__ ((section(".weightmem_sram"))) '
            weightstr += f"unsigned char {node.name}_weights[{len(weights)}] = "
            weightstr += "{"
            weightstr += ", ".join("0x"+format(x, '02x') for x in weights)
            weightstr += "};\r\n"

            for const in constants[1:]:
                if const != 0:
                    val = bytes(getattr(node,const)['value'])
                    weightstr += 'PI_L2 '
                    weightstr += f"unsigned char {node.name}_{const}[{len(val)}] = "
                    weightstr += "{"
                    weightstr += ", ".join("0x"+format(x, '02x') for x in val)
                    weightstr += "};\r\n"

            weightstr_h = f"#ifndef __INCLUDE_GUARD_{node.name}\r\n"
            weightstr_h += f"#define __INCLUDE_GUARD_{node.name}\r\n"
            weightstr_h += f"extern unsigned char {node.name}_weights[{len(weights)}];\r\n"
            for const in constants[1:]:
                if const != 0:
                    val = bytes(getattr(node,const)['value'])
                    weightstr_h += f"extern unsigned char {node.name}_{const}[{len(val)}];\r\n"
            weightstr_h += f"\r\n#endif"

            filepath = os.path.join(self.app_directory, 'src', node.prefix + node.name + "_weights.c")
            with open(filepath, 'w') as file:
                file.write(weightstr)

            filepath = os.path.join(self.app_directory, 'inc', node.prefix + node.name + "_weights.h")
            with open(filepath, 'w') as file:
                file.write(weightstr_h)

# C_Parser.py
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
from collections import OrderedDict
import numpy as np
from mako.template import Template

# DORY modules
from dory.Hardware_targets.neureka.nnx_C_Parser import nnx_C_Parser
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
import dory.Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
from dory.Hardware_targets.neureka.neureka.Neureka import Neureka
import dory.Utils.Templates_writer.writer_utils as utils

class C_Parser_Siracusa(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory, n_inputs=1, prefix=''):

        file_path = self.get_file_path()
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", app_directory, n_inputs, prefix)
        self.acc = Neureka()
        try:
            db = HW_description['double_buffering']
        except KeyError:
            print("C_Parser_Siracusa: Key 'double_buffering' not found in HW_description.json - setting to 2")
            db = 2
        self.double_buffering = db

        self.weights_names = []
        self.weights_vectors = []
        self.weights_dimensions = []

    def get_file_path(self):
        raise NotImplementedError("To be implemented by child class!")

    def copy_backend_files(self, node):
        if self.precision_library == 'auto':
            self.precision_library = '8bit'
            if "Addition" not in node.name and "Pool" not in node.name:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8 or node.get_parameter('weight_bits') < 8:
                    self.precision_library = 'mixed-sw'
            else:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8:
                    self.precision_library = 'mixed-sw'

        root = self.get_file_path()
        if self.precision_library == "8bit":
            files = os.path.join(root, "../Backend_Kernels/pulp-nn/")
        elif self.precision_library == "mixed-sw":
            files = os.path.join(root, "../Backend_Kernels/pulp-nn-mixed/XpulpV2/")
        elif self.precision_library == "mixed-hw":
            files = os.path.join(root, "../Backend_Kernels/pulp-nn-mixed/XpulpNN/")
        if os.listdir(os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library)))[0] not in os.listdir(self.inc_dir):
            for file in os.listdir(os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, self.inc_dir_rel)))
        if self.precision_library == "8bit":
            if os.listdir(os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.app_directory, self.src_dir_rel)):
                for file in os.listdir(os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library))):
                    file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
                    os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, self.src_dir_rel)))
        elif self.precision_library in ["mixed-sw", "mixed-hw"]:
            Input_bits = str(node.get_parameter('input_activation_bits'))
            Output_bits = str(node.get_parameter('output_activation_bits'))
            Input_type = node.get_parameter('input_activation_type')[0]
            Output_type = node.get_parameter('output_activation_type')[0]
            out = "_" + Output_type + Output_bits
            in_out = "_" + Input_type + Input_bits + out
            maybe_x = 'x' if self.precision_library == "mixed-hw" else ''
            if "Addition" in node.name:
                in1_in2_out = "_" + Input_type + Input_bits + "_" + node.get_parameter('second_input_activation_type')[0] + str(node.get_parameter('second_input_activation_bits')) + "_" + Output_type + Output_bits
                file = f'Add/{maybe_x}pulp_nn_add{in1_in2_out}.c'
            elif "Pool" in node.name and "Max" in node.op_type:
                file = f'Pooling/MaxPool/{maybe_x}pulp_nn_maxpool{out}.c'
            elif "Pool" in node.name and ("Avg" in node.op_type or "Average" in node.op_type):
                file = f'Pooling/AvgPool/{maybe_x}pulp_nn_avgpool{in_out}.c'

            in_out_weights = "_" + Input_type + Input_bits + "_" + Output_type + Output_bits + "_" + node.get_parameter('weight_type')[0] + str(node.get_parameter('weight_bits'))
            if "Conv" in node.name and node.group > 1:
                file = f'Depthwise/{maybe_x}pulp_nn_depthwise{in_out_weights}.c'
            elif "Conv" in node.name and node.group == 1:
                if node.conv1d and self.precision_library == 'mixed-hw':
                    file = f'Convolution/xpulp_nn_conv1d{in_out_weights}.c'
                else:
                    file = f'Convolution/{maybe_x}pulp_nn_conv{in_out_weights}.c'
            elif "FullyConnected" in node.name and node.output_activation_bits == 32:
                file = f'LinearNoQuant/{maybe_x}pulp_nn_linear{in_out_weights}.c'
            elif "FullyConnected" in node.name:
                file = f'LinearQuant/{maybe_x}pulp_nn_linear{in_out_weights}.c'
            file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
            os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, self.src_dir_rel)))
            if ("Conv" in node.name or "FullyConnected" in node.name) and node.get_parameter('output_activation_bits') != 32:
                in_bits_matmul = "8" if self.precision_library == "mixed-sw" else str(Input_bits)
                in_out_weights = "_" + Input_type + in_bits_matmul + "_" + Output_type + Output_bits + "_" + node.get_parameter('weight_type')[0] + str(node.get_parameter('weight_bits'))
                file = f'MatrixMultiplication/{maybe_x}pulp_nn_matmul{in_out_weights}.c'
                file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, self.src_dir_rel)))


    def map_layer_to_C_file(self, node, n_memory_levels, tmpl_dir, out_dir):
        self.copy_backend_files(node)
        if n_memory_levels > 2 and (node.L3_input != 0 or (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) or (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"])):
            Layer2D_writer.print_template_layer_L3(node, tmpl_dir, out_dir)
            if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                node.tiling_dimensions["L2"]["output_dimensions"][1]  = int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - node.kernel_shape[0] + node.strides[0]) / node.strides[0]))
            if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                node.tiling_dimensions["L2"]["input_dimensions"][1]   = node.tiling_dimensions["L2"]["output_dimensions"][1] * node.strides[0] + node.kernel_shape[0] - node.strides[0]
            node.name = node.name + "_L2"
            padding = node.pads
            node.pads = [0, padding[1], 0, padding[3]]
            Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering=self.double_buffering)
            node.name = node.name[:-3]
            if padding[0] > 0:
                node.name = node.name + "_L2_p_t"
                node.pads = [padding[0], padding[1], 0, padding[3]]
                Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering=self.double_buffering)
                node.name = node.name[:-1] + "b"
                node.pads = [0, padding[1], padding[2], padding[3]]
                node.tiling_dimensions["L2"]["input_dimensions"][1] -= (padding[2] - ((node.tiling_dimensions["L3"]["input_dimensions"][1] + padding[0] + padding[2]) - (node.tiling_dimensions["L3"]["output_dimensions"][1]* node.strides[0] + node.kernel_shape[0] - node.strides[0])))
                if node.tiling_dimensions["L1"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                    node.tiling_dimensions["L1"]["input_dimensions"][1] = node.tiling_dimensions["L2"]["input_dimensions"][1]
                if node.tiling_dimensions["L1"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                    node.tiling_dimensions["L1"]["output_dimensions"][1] = node.tiling_dimensions["L2"]["output_dimensions"][1]
                Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering=self.double_buffering)
                node.name = node.name[:-7]
        else:
            if node.tiling_dimensions["L2"]["input_dimensions"][2] == node.tiling_dimensions["L1"]["input_dimensions"][2]:
                node.tiling_dimensions["L1"]["output_dimensions"][2] = int((node.tiling_dimensions["L1"]["input_dimensions"][2] + (node.pads[1] + node.pads[3]) - node.kernel_shape[1] + node.strides[1]) / node.strides[1])
            if node.tiling_dimensions["L2"]["input_dimensions"][1] == node.tiling_dimensions["L1"]["input_dimensions"][1]:
                node.tiling_dimensions["L1"]["output_dimensions"][1] = int((node.tiling_dimensions["L1"]["input_dimensions"][1] + (node.pads[0] + node.pads[2]) - node.kernel_shape[0] + node.strides[0]) / node.strides[0])
            Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering=self.double_buffering)

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.realpath(os.path.join(self.get_file_path(), 'Templates/layer_templates'))
        out_dir = self.app_directory
        n_memory_levels = self.HW_description['memory']['levels']

        for i, node in enumerate(self.HWgraph):
            node.prefix = self.prefix
            if not hasattr(node, "offloadable") or not node.offloadable:
                self.map_layer_to_C_file(node, n_memory_levels, tmpl_dir, out_dir)
            else:
                nnx_C_Parser.copy_backend_files(node, self.app_directory, self.config_file["nnx_dir"], "neureka")
                nnx_C_Parser.map_layer_to_C_file(node, self.config_file, self.acc, os.path.join(self.config_file['nnx_dir'], "Templates/layer_templates"), out_dir, self.HW_description)

    def create_hex_weight(self, node):
        if (node.HW_description['memory']['levels'] > 2) and (not hasattr(node, "offloadable") or not node.offloadable):
            super().create_hex_weight(node)
        elif hasattr(node, "offloadable") and node.offloadable:

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
            weightstr += f"#include \"{node.prefix}{node.name}_weights.h\"\r\n"
            weightstr += f"#include \"pmsis.h\"\r\n"
            weightstr += '__attribute__ ((section(".weightmem_sram"))) '
            weightstr += f"unsigned char {node.prefix}{node.name}_weights[{len(weights)}] = "
            weightstr += "{"
            weightstr += ", ".join("0x"+format(x, '02x') for x in weights)
            weightstr += "};\r\n"

            for const in constants[1:]:
                if const != 0:
                    val = bytes(getattr(node,const)['value'])
                    weightstr += 'PI_L2 '
                    weightstr += f"unsigned char {node.prefix}{node.name}_{const}[{len(val)}] = "
                    weightstr += "{"
                    weightstr += ", ".join("0x"+format(x, '02x') for x in val)
                    weightstr += "};\r\n"

            weightstr_h = f"#ifndef __INCLUDE_GUARD_{node.prefix}{node.name}\r\n"
            weightstr_h += f"#define __INCLUDE_GUARD_{node.prefix}{node.name}\r\n"
            weightstr_h += f"extern unsigned char {node.prefix}{node.name}_weights[{len(weights)}];\r\n"
            for const in constants[1:]:
                if const != 0:
                    val = bytes(getattr(node,const)['value'])
                    weightstr_h += f"extern unsigned char {node.prefix}{node.name}_{const}[{len(val)}];\r\n"
            weightstr_h += f"\r\n#endif"

            filepath = os.path.join(self.app_directory, 'src', node.prefix + node.name + "_weights.c")
            with open(filepath, 'w') as file:
                file.write(weightstr)

            filepath = os.path.join(self.app_directory, 'inc', node.prefix + node.name + "_weights.h")
            with open(filepath, 'w') as file:
                file.write(weightstr_h)
        else:
            print("\nGenerating .h weight files.")

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
            weights = np.asarray([])
            for i in np.arange(4):
                if constants[i]!= 0:
                    weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))
            while len(weights) % 4 != 0:
                weights = np.concatenate((weights, np.asarray([0])))
            ww, ww_dim = utils.print_test_vector(weights, 'char'), weights.shape[0]
            self.weights_names.append(node.name)
            self.weights_vectors.append(ww)
            self.weights_dimensions.append(ww_dim)
            tk = OrderedDict([])
            tk['weights_names'] = self.weights_names
            tk['weights_vectors'] = self.weights_vectors
            tk['weights_dimensions'] = self.weights_dimensions
            tk['DORY_HW_graph'] = self.HWgraph
            tk['prefix'] = node.prefix
            tk['sdk'] = node.HW_description["software development kit"]["name"]
            root = os.path.dirname(__file__)
            tmpl = Template(filename=os.path.join(root, "Templates/weights_h_template.h"))
            s = tmpl.render(**tk)
            save_string = os.path.join(self.inc_dir, f'{node.prefix}weights.h')
            with open(save_string, "w") as f:
                f.write(s)
            tmpl = Template(filename=os.path.join(root, "Templates/weights_definition_h_template.h"))
            s = tmpl.render(**tk)
            save_string = os.path.join(self.inc_dir, f'{node.prefix}weights_definition.h')
            with open(save_string, "w") as f:
                f.write(s)

    def create_hex_input(self):
        if (self.HW_description['memory']['levels'] > 2):
            return super().create_hex_input()

        print("\nGenerating .h input file.")
        x_in_l = []
        for in_idx in range(self.n_inputs):
            infile = 'input.txt' if self.n_inputs == 1 else f'input_{in_idx}.txt'
            try:
                x_in = np.loadtxt(os.path.join(self.network_directory, infile), delimiter=',', dtype=np.int16, usecols=[0])
            except FileNotFoundError:
                print(f"========= WARNING ==========\nInput file {os.path.join(self.network_directory, 'input.txt')} not found; generating random inputs!")
                x_in = np.random.randint(low=0, high=2*8,size=self.group * self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],dtype=np.int16)
            x_in_l.append(x_in.flatten())

        x_in = np.concatenate(x_in_l)
        in_node = self.HWgraph[0]
        in_bits = in_node.input_activation_bits
        if in_bits != 8:
            x_in = HW_node._compress(x_in, in_bits)

        temp = x_in
        # input_values = utils.print_test_vector(temp.flatten(), 'char')
        s = ''
        for num in temp:
            if num > 0:
                s += f"{hex(np.uint8(num))}, "
            else:
                s += f"{hex(np.uint8(num+256))}, "
        tk = OrderedDict([])
        tk['input_values'] = s[:-2]
        tk['prefix'] = self.prefix
        tk['dimension'] = len(x_in)
        tk['sdk'] = self.HW_description["software development kit"]["name"]
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/input_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, f'{self.prefix}input.h')
        with open(save_string, "w") as f:
            f.write(s)

     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
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
import numpy as np
import json
import os

# DORY modules
from dory.Parsers import HW_node, Layer_node
from dory.Parsers import HW_node_neureka
from dory.Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from functools import partial
from dory.Hardware_targets.neureka.nnx_HW_Parser import nnx_HW_Parser
from dory.Hardware_targets.neureka.neureka.Neureka import Neureka

from dory.Hardware_targets.neureka.Tiler.tiler import Tiler as nTiler
from functools import partial

class onnx_manager_Siracusa(Parser_DORY_to_HW):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, n_inputs=1):
        layers_supported_by_HW_Backend_IR = ["Convolution", "Pooling", "FullyConnected", "Addition", "QAddition"]
        layers_supported_by_HW_Backend_IR+= ["ReluConvolution", "ReluPooling", "ReluFullyConnected", "ReluAddition", "ReluQAddition"]
        layers_supported_by_HW_Backend_IR+= ["BNReluConvolution", "RequantPooling", "BNReluFullyConnected", "BNReluAddition", "BNReluQAddition"]
        file_path = self.get_file_path()
        pattern_rewriter = self.get_pattern_rewriter()
        with open(os.path.join(file_path, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)

        try:
            db = HW_description['double_buffering']
        except KeyError:
            print("onnx_manager_Siracusa: Key 'double_buffering' not found in HW_description.json - setting to 2")
            db = 2

        self.double_buffering = db

        self.acc = Neureka()
        tiler = self.get_tiler()
        #HACK GEORGR:
        # keep the unified Tiler interface but pass the double_buffering
        # parameter correctly by pre-supplying the argument
        tiler = partial(tiler, double_buffering=self.double_buffering)

        super().__init__(graph, rules, pattern_rewriter, layers_supported_by_HW_Backend_IR, HW_description,
                         os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), config_file, tiler, n_inputs)

    def get_file_path(self):
        raise NotImplementedError("To be implemented by child class!")

    def get_pattern_rewriter(self):
        raise NotImplementedError("To be implemented by child class!")

    def get_tiler(self):
        raise NotImplementedError("To be implemented by child class!")

    def adjust_data_layout_node(self, i, node):
        if "FullyConnected" in node.name:
            for name in node.constant_names:
                if name not in ["l","k","outshift","outmul"]:
                    if "bias" not in name:
                        weights_name = name
            if node.__dict__[weights_name]["layout"] == "CinCout":
                node.__dict__[weights_name]["value"] = node.__dict__[weights_name]["value"].T
                node.__dict__[weights_name]["layout"] = "CoutCin"
            if i != 0 and self.DORY_Graph[i-1].layout == "CHW":
                temp = node.__dict__[weights_name]["value"]
                prev_node = self.DORY_Graph[i-1]
                temp = temp.reshape(node.output_channels, prev_node.output_channels, prev_node.output_dimensions[0], prev_node.output_dimensions[1])
                temp = np.transpose(temp, (0, 2, 3, 1))
                temp = temp.flatten()
                node.__dict__[weights_name]["value"] = temp
                # needed to compute final checksum for <8b layers
        elif "Convolution" in node.name:
            for name in node.constant_names:
                if name not in ["l","k","outshift","outmul"]:
                    if "bias" not in name:
                        weights_name = name
            if node.__dict__[weights_name]["layout"] == "CoutCinK":
                if node.conv1d:
                    node.__dict__[weights_name]["value"] = node.__dict__[weights_name]["value"][:,:,None,:]
                transp = (0,2,3,1)
                node.__dict__[weights_name]["value"] = np.transpose(node.__dict__[weights_name]["value"], transp)
                node.__dict__[weights_name]["layout"] = "CoutKCin"


    def adjust_data_layout(self):
        print("\nGAP8 Backend: Adjusting Data Layout to HWC and CoutKCin.")
        for i, node in enumerate(self.DORY_Graph):
            if hasattr(node, "offloadable") and node.offloadable:
                # SCHEREMO: Offload to N-EUREKA
                nnx_HW_Parser.adjust_data_layout_node(node, self.acc)
            else:
                self.adjust_data_layout_node(i, node)

    def check_parameters(self):
        WARNINGS =0
        for node in self.DORY_Graph:
            for key, value in node.__dict__.items():
                if key not in HW_node.HW_node(Layer_node.Layer_node(), self.HW_description).__dict__.keys() and key not in Layer_node.Layer_node().__dict__.keys():
                    if key not in node.constant_names:
                        print("WARNING: DORY Backend. Attribute {} of Node {} is not inside the predefined parameters for DORY nodes.".format(key, node.name))
                        WARNINGS +=1
                if isinstance(value,list):
                    if len(value) == 0:
                        WARNINGS +=1
                        print("WARNING: DORY Backend. Attribute {} of Node {} is an empty list.".format(key, node.name))
                if isinstance(value,type(None)):
                    WARNINGS +=1
                    print("WARNING: DORY Backend. Attribute {} of Node {} is still not initialized.".format(key, node.name))
        print("\nDORY checking of the attribute of the graph: {} WARNINGS\n".format(WARNINGS))

    @staticmethod
    def is_offloadable(node: Layer_node) -> bool:
        #SCHEREMO: Check if it's an 8-Bit x 8-Bit or lower convolution
        if node.op_type == "BNReluConv" and node.weight_bits == 8 and node.input_activation_bits == 8 and node.output_activation_type == 'uint' and node.input_activation_type == 'uint' and node.input_channels <= 192 and node.output_channels <= 192:
            #SCHEREMO: Check if it's a pointwise convolution:
            if node.group == 1 and node.kernel_shape == [1,1]:
                print("Offloading to NEUREKA...")
                return True
            #SCHEREMO: Check if it's a depthwise 3x3 convolution:
            elif node.input_channels == node.output_channels and node.group == node.output_channels and node.kernel_shape == [3,3]:
                return False

        return False


    #SCHEREMO: Define offloading to N-EUREKA
    def mapping_to_HW_nodes(self):
        super().mapping_to_HW_nodes()
        #SCHEREMO: function hooks that check if a node is offloadable to N-EUREKA and if so, mark it.
        if 'offload' in self.config_file and self.config_file['offload'] == True:
            print("Offloading to N-EUREKA")
            for idx, node in enumerate(self.DORY_Graph):
                if idx == 0:
                    node.offloadable = False
                else:
                    node.offloadable  = onnx_manager_Siracusa.is_offloadable(node)

    def tile_node(self, i, node_to_tile, previous_node):
        if hasattr(node_to_tile, "offloadable") and node_to_tile.offloadable:
            New_HW_node = HW_node_neureka.HW_node_neureka(node_to_tile, self.HW_description, self.acc)
        else:
            New_HW_node = HW_node.HW_node(node_to_tile, self.HW_description)
        if hasattr(New_HW_node, "offloadable") and New_HW_node.offloadable:
            New_HW_node.Tiler = partial(nTiler, conf=self.config_file, acc=self.acc)
            ws = self.acc.weights_size
        if i > 0:
            New_HW_node.create_tiling_dimensions(previous_node, self.config_file)
        else:
            New_HW_node.create_tiling_dimensions(New_HW_node, self.config_file)
        if hasattr(New_HW_node, "offloadable") and New_HW_node.offloadable:
            New_HW_node.Tiler = None
            New_HW_node.acc = None
        return New_HW_node

    def tiling(self):
        print("\nInsert tiling parameters per layer inside graph nodes")
        previous_node = None
        for i, node_to_tile in enumerate(self.DORY_Graph):
            New_HW_node = self.tile_node(i, node_to_tile, previous_node)
            previous_node = New_HW_node
            self.DORY_Graph[i] = New_HW_node

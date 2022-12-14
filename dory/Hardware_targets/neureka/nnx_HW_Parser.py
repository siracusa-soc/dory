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
import os
import json

# DORY modules
from dory.Parsers.HW_node_neureka import HW_node
from dory.Parsers.Layer_node import Layer_node
from dory.Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from .HW_Pattern_rewriter import Pattern_rewriter
from .Tiler.tiler import Tiler

from functools import partial

class nnx_HW_Parser(Parser_DORY_to_HW):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, confdir, accelerator):
        supported_layers = ["Convolution", "ReluConvolution", "BNReluConvolution"]
        self.nnxdir = os.path.dirname(__file__)
        with open(os.path.join(self.nnxdir, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(self.nnxdir, "HW_description.json")) as f:
            hw_description = json.load(f)
        self.acc = accelerator
        weights_size = self.acc.weights_size
        Tiler.acc = self.acc
        super().__init__(graph, rules, Pattern_rewriter, supported_layers, hw_description,
                         os.path.join(confdir, os.path.dirname(conf["onnx_file"])), conf, Tiler)

    @staticmethod
    def adjust_data_layout_node(node, acc):
        if "Convolution" in node.name:
            for name in node.constant_names:
                if name not in ["l", "k", "outshift", "outmul"] and "bias" not in name:
                    weights_name = name
            weights = getattr(node, weights_name)

            weights["value"] = acc.conv_unroll(weights["value"] + (2**(node.weight_bits-1)), node.weight_bits, weights["layout"], node.group > 1)


    def adjust_data_layout(self):
        print("\nNNX Backend: Adjusting Feature Data Layout to HWC and Weights Data Layout to accelerator specific")
        for i, node in enumerate(self.DORY_Graph):
            nnx_HW_Parser.adjust_data_layout_node(node, self.acc)

    def check_parameters(self):
        warning_count = 0

        def warning(msg):
            print(f'WARNING: DORY Backend. Attribute {attr} of Node {node.name} is {msg}.')
            nonlocal warning_count
            warning_count += 1

        vanilla_attrs = list(Layer_node().__dict__.keys()) + \
                        list(HW_node(Layer_node(), self.HW_description, self.acc).__dict__.keys())

        for node in self.DORY_Graph:
            for attr, value in node.__dict__.items():
                if attr not in vanilla_attrs and attr not in node.constant_names:
                    warning('not inside the predefined parameters for DORY nodes')
                if value is None:
                    warning('not initialized')
                elif isinstance(value, list) and len(value) == 0:
                    warning('an empty list')

        print(f"\nDORY checking of the attribute of the graph: {warning_count} warnings\n")

    def tile_node(self, i, node_to_tile, previous_node):
        New_HW_node = HW_node(node_to_tile, self.HW_description, self.acc)
        New_HW_node.Tiler = partial(Tiler, conf=self.config_file, acc=self.acc)
        ws = self.acc.weights_size
        if i > 0:
            New_HW_node.create_tiling_dimensions(previous_node, self.config_file)
        else:
            New_HW_node.create_tiling_dimensions(New_HW_node, self.config_file)
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

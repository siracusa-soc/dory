     # should work even without -*-
#
# tiling.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Francesco Conti <f.conti@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
#
# Copyright (C) 2018-2020 University of Bologna
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

# tilers for layers
from .tiler_conv2d import Tiler_Conv2D
from .tiler_pool2d import Tiler_Pool2D
from ortools.constraint_solver import pywrapcp


class Tiler:
    # Class to generate the Tiling of the layer.
    acc = None

    def __init__(self, node, prev_node, conf):
        self.node = node
        self.prev_node = prev_node
        self.conf = conf

    def get_tiling(self, level):
        # This function is used to create the tiling of either a convolutional layer or
        # a fully connected or a pooling layer. The relu is included automatically in conv/FC.
        if 'Conv' in self.node.name or 'FullyConnected' in self.node.name:
            return Tiler_Conv2D(self.node, self.prev_node, self.conf, self.acc).get_tiling(level)
        elif 'Pooling' in self.node.name:
            tiler = Tiler_Pool2D(self)
            # Scheremo: The inheritance scheme and initialization scheme of tilers is unclear.
            # I am working around this by explicitly initializing, which is definitely bad practice.
            # There should probably be a clear inheritance / polymorphism strategy for tilers.
            tiler.HW_node = self.node
            tiler.previous_HW_node = self.prev_node
            tiler.conf = self.conf
            tiler.code_reserved_space = self.conf['code reserved space']
            return tiler.get_tiling(level)
        else:
            print("Not supported Layer.")
            return None

# should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# PULP_node.py
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
import copy
import os

# DORY modules
from .HW_node import HW_node


class HW_node(HW_node):
    # A self allocated in the PULP_Graph

    # Class attributes
    Tiler = None

    def __init__(self, node, HW_description, acc):
        super().__init__(node, HW_description)
        self.acc = acc

    def weights_size(self, weights_dim):
        return self.acc.weights_size(weights_dim[0], weights_dim[1], self.kernel_shape, self.weight_bits, self.group>1)

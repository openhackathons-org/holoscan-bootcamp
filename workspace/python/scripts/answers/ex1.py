# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

class MyOp(Operator):

    def __init__(self, *args, **kwargs):
        self.index = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.input("in3")
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        
        # Always emit in1 value in out1 port
        in1_value = op_input.receive("in1")
        op_output.emit(in1_value, "out1")

        # each input needs to be received, regardless of utilization
        # Even if in2 and in3 won't be utilized at the same time,
        # they still both need to be be received
        in2_value = op_input.receive("in2")
        in3_value = op_input.receive("in3")
        
        # If tick is even
        if self.index % 2 == 0:   
            op_output.emit(in2_value, "out2")
        else:
            op_output.emit(in3_value, "out2")
        
        self.index += 1

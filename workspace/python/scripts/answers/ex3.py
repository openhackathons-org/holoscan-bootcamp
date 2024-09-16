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

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

import cupy as cp


class SourceOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        op_output.emit(cp.random.randn(32768), "out")

        
class FFTOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        sig = op_input.receive("in")
        op_output.emit(cp.fft.fft(sig), "out")

        
class SinkOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        
    def compute(self, op_input, op_output, context):
        sig = op_input.receive("in")
        print(sig)


class FFTApp(Application):
    def compose(self):
        src = SourceOp(self, CountCondition(self, 3), name="src_op")
        fft = FFTOp(self, name="fft_op")
        sink = SinkOp(self, name="sink_op")

        # Connect the operators into the workflow:  src -> fft -> sink
        self.add_flow(src, fft)
        self.add_flow(fft, sink)
        

if __name__ == "__main__":
    app = FFTApp()
    app.config("")
    app.run()
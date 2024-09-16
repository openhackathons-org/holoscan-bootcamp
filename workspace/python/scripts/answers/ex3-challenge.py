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
    def __init__(self, *args, **kwargs):
        self.rng = cp.random.default_rng()
        self.static_out = self.rng.standard_normal((1000, 1000), dtype=cp.float32)
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.output("static_out")
        spec.output("variable_out")
        
    def compute(self, op_input, op_output, context):
        op_output.emit(self.rng.standard_normal((1000, 1000), dtype=cp.float32), "variable_out")
        op_output.emit(self.static_out, "static_out")

        
class MatMulOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.input("in_static")
        spec.input("in_variable")
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        mat_static = op_input.receive("in_static")
        mat_dynamic = op_input.receive("in_variable")
        op_output.emit(cp.matmul(mat_static, mat_dynamic), "out")

        
class SinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        
    def compute(self, op_input, op_output, context):
        sig = op_input.receive("in")
        print(sig)


class MatMulApp(Application):
    def compose(self):
        src = SourceOp(self, CountCondition(self, 5), name="src_op")
        matmul = MatMulOp(self, name="matmul_op")
        sink = SinkOp(self, name="sink_op")

        # Connect the operators into the workflow:  src -> matmul -> sink
        self.add_flow(src, matmul, {("static_out", "in_static"), ("variable_out", "in_variable")})
        self.add_flow(matmul, sink)
        

if __name__ == "__main__":
    app = MatMulApp()
    app.config("")
    app.run()
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

# For more information and the code refer to the github page for tao_peoplenet example on holohub:
# https://github.com/nvidia-holoscan/holohub/tree/main/applications/tao_peoplenet

import os
from argparse import ArgumentParser

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator
from holoscan.schedulers import GreedyScheduler


class PreprocessorOp(Operator):
    """Operator to format input image for inference"""
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed")).get()
        # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)

        # Create output message
        out_message = {"preprocessed": tensor}
        op_output.emit(out_message, "out")


class PostprocessorOp(Operator):
    """Operator to post-process inference output:
    * Reparameterize bounding boxes
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.15)
        spec.param("score_threshold", 0.5)
        spec.param("image_width", None)
        spec.param("image_height", None)
        spec.param("box_scale", None)
        spec.param("box_offset", None)
        spec.param("grid_height", None)
        spec.param("grid_width", None)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Convert input to cupy array
        boxes = cp.asarray(in_message.get("boxes"))[0, ...]
        scores = cp.asarray(in_message.get("scores"))[0, ...]

        # PeopleNet has three classes:
        # 0. Person
        # 1. Bag
        # 2. Face
        # Here we only keep the Person and Face classes
        boxes = boxes[[0, 1, 2, 3, 8, 9, 10, 11], ...][None]
        scores = scores[[0, 2], ...][None]

        # Loop over label classes
        out = {"person": None, "faces": None}
        for i, label in enumerate(out):
            # Reparameterize boxes
            out[label], scores_nms = self.reparameterize_boxes(
                boxes[:, 0 + i * 4 : 4 + i * 4, ...],
                scores[:, i, ...][None],
            )

            # Non-max suppression
            out[label], _ = self.nms(out[label], scores_nms)

            # Reshape for HoloViz
            if len(out[label]) == 0:
                out[label] = np.zeros([1, 2, 2]).astype(np.float32)
            else:
                out[label][:, [0, 2]] /= self.image_width
                out[label][:, [1, 3]] /= self.image_height
                out[label] = cp.reshape(out[label][None], (1, -1, 2))
                # out[label] = cp.asnumpy(out[label])

        # Create output message
        op_output.emit(out, "out")

    def nms(self, boxes, scores):
        """Non-max suppression (NMS)

        Parameters
        ----------
        boxes : array (4, n)
        scores : array (n,)

        Returns
        ----------
        boxes : array (m, 4)
        scores : array (m,)

        """
        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, scores_out = [], []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]

            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            scores_out.append(scores[index])

            # Get coordinates
            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            # Compute IOU
            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            # Threshold and prune
            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        # To array
        boxes = cp.asarray(boxes_out)
        scores = cp.asarray(scores_out)

        return boxes, scores

    def reparameterize_boxes(self, boxes, scores):
        """Reparameterize boxes from corner+width+height to corner+corner.

        Parameters
        ----------
        boxes : array (1, 4, grid_height, grid_width)
        scores : array (1, 1, grid_height, grid_width)

        Returns
        ----------
        boxes : array (4, n)
        scores : array (n,)

        """
        cell_height = self.image_height / self.grid_height
        cell_width = self.image_width / self.grid_width

        # Generate the grid coordinates
        mx, my = cp.meshgrid(cp.arange(self.grid_width), cp.arange(self.grid_height))
        mx = mx.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))
        my = my.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))

        # Compute the box corners
        xmin = -(boxes[0, 0, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymin = -(boxes[0, 1, ...] + self.box_offset) * self.box_scale + my * cell_height
        xmax = (boxes[0, 2, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymax = (boxes[0, 3, ...] + self.box_offset) * self.box_scale + my * cell_height
        boxes = cp.concatenate([xmin, ymin, xmax, ymax], axis=1)

        # Select the scores that are above the threshold
        scores_mask = scores > self.score_threshold
        scores = scores[scores_mask]
        scores_mask = cp.repeat(scores_mask, 4, axis=1)
        boxes = boxes[scores_mask]

        # Reshape after masking
        n = int(boxes.size / 4)
        boxes = boxes.reshape(4, n)

        return boxes, scores


class PeopleAndFaceDetectApp(Application):
    def __init__(self, data_path, model_path, *args, **kwargs):
        """Initialize the face and people detection application"""
        super().__init__(*args, **kwargs)
        self.name = "People and Face Detection App"
        self.sample_data_path = data_path
        self.model_path = model_path

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        # Video source operator
        source = VideoStreamReplayerOp(
            self,
            name="replayer_source",
            directory=self.sample_data_path,
            **self.kwargs("replayer_source"),
        )

        # Format converter operator
        preprocessor_args = self.kwargs("preprocessor")
        format_converter = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )

        # Preprocessor operator
        preprocessor = PreprocessorOp(
            self,
            name="transpose",
            pool=pool,
        )

        # Inference operator
        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {
            "face_detect": os.path.join(self.sample_data_path, "resnet34_peoplenet_int8.onnx")
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_args,
        )

        # Postprocessor operator
        postprocessor_args = self.kwargs("postprocessor")
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        # Vizualization operator
        holoviz = HolovizOp(self,
                            allocator=pool,
                            name="holoviz",
                            headless=True, # this True to run the app on the cluster (see below)
                            **self.kwargs("holoviz"))


        self.add_flow(source, holoviz, {("output", "receivers")})
        self.add_flow(source, format_converter)
        self.add_flow(format_converter, preprocessor)
        self.add_flow(preprocessor, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, holoviz, {("out", "receivers")})


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "tao_peoplenet.yaml")
    data_path = os.path.join(os.path.dirname(__file__), "data/")
    model_path = os.path.join(os.path.dirname(__file__), "data/resnet34_peoplenet_int8.onnx")

    app = PeopleAndFaceDetectApp(data_path, model_path)
    app.config(config_file)
    scheduler = GreedyScheduler(app, name="greedy_scheduler", max_duration_ms=5000)
    app.scheduler(scheduler)

    app.run()
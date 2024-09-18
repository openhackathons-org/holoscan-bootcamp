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

# To build the docker container, run: $ docker build . -t holoscan:v2.3.0-dgpu -f Dockerfile --network host
# To run: $ docker run -it --rm --ipc=host --runtime=nvidia --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888 holoscan:v2.3.0-dgpu
# Finally, open http://127.0.0.1:8888/

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v2.3.0-dgpu

ARG DEBIAN_FRONTEND=noninteractive

#ENV VK_LOADER_DEBUG="all"

RUN pip3 install jupyterlab

RUN apt update
RUN apt install --no-install-recommends -y ffmpeg
RUN apt install libegl1

COPY workspace/ /workspace/

RUN ffmpeg -i /workspace/python/scripts/tao_peoplenet/data/people.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python bin/convert_video_to_gxf_entities.py --directory /workspace/python/scripts/tao_peoplenet/data/ --basename people --width 1920 --height 1080 --framerate 30


#CMD ipython
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python

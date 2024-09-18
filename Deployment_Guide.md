# NVIDIA Holoscan Deployment Guide
This Bootcamp focuses on building an end-to-end AI-enabled streaming pipeline using the Holoscan SDK, handling sensor I/O, applying a trained AI model to a real time sensor stream, and building GPU accelerated applications. We will also discuss techniques to measure application performance and transition from prototype to production.

## Deploying the materials

### Prerequisites
To run this tutorial, you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create an NGC account and generate an [API key](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

The material has been tested to be working with NVIDIA A100 GPU, please contact us if you require assistance in deploying the content.


### Tested environment

These materials was tested with both Docker and Singularity on an NVIDIA A100 GPU in an x86-64 platform installed with a driver version of `535.104.05`.

### Deploying with container

These materials can be deployed with either Docker or Singularity container, refer to the respective sections for the instructions.

#### Docker Container

- To build the docker container for NVIDIA Holoscan bootcamp, follow the below steps:

  1. `sudo docker build . -t holoscan:openhackathons -f Dockerfile --network host`
  2. `sudo docker run -it --rm --ipc=host --runtime=nvidia --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888 holoscan:openhackathons`
  3. Now, open the jupyter lab in browser: http://localhost:8888, and start working on the lab by clicking on the `Python-Holoscan-Tutorial.ipynb` notebook

#### Docker Container using rootless Docker
- To build the container using rootless Docker on a cluster (e.g. Curiosity) use Dockerfile_cluster:

  1. `sudo docker build . -t holoscan:openhackathons -f Dockerfile_cluster --network host`
  2. `sudo docker run -it --rm --ipc=host --runtime=nvidia --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888 holoscan:openhackathons`
  3. Now, open the jupyter lab in browser: http://localhost:8888, and start working on the lab by clicking on the `Python-Holoscan-Tutorial.ipynb` notebook

#### Singularity Container

- To build the singularity container for NVIDIA Holoscan bootcamp, follow the below steps:

  1. `singularity build --fakeroot --sandbox holoscan:openhackathons singularity`
  3. `singularity run --writable --nv holoscan:openhackathons jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python`
  3. Now, open the jupyter lab in browser: http://localhost:8888, and start working on the lab by clicking on the `Python-Holoscan-Tutorial.ipynb` notebook


### Known issues

- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/openhackathons-org/holoscan-bootcamp/issues).
FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN python3 -m pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.4

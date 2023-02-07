ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# install dependencies
RUN pip install python-dotenv tqdm wandb
RUN pip install black flake8 isort pre-commit

ENV PYTHONPATH=/work

WORKDIR /work

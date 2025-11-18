# Dockerfile to build CI/CD environment

# Build from an image containing needed cuda libs
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, create venv, install pip and tox in it
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-distutils && \
    python3.11 -m venv /venv && \
    /venv/bin/pip install -U pip tox

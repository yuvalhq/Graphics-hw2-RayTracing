FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jerusalem

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt update \
    && apt install -y python3.6 \
    python3.6-dev \
    python3-pip \
    python3-attr \
    python3-pil:amd64 \
    python3-pil.imagetk:amd64 \
    python3-numpy \
    git \
    && pip3 install ipdb ipython black isort mypy flake8 pylint

COPY . /usr/src
WORKDIR /usr/src

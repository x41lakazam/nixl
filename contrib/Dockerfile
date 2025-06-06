# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
ARG BASE_IMAGE_TAG="25.03-cuda12.8-devel-ubuntu24.04"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

ARG DEFAULT_PYTHON_VERSION="3.12"

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ninja-build \
    pybind11-dev \
    libclang-dev \
    cmake \
    libgflags-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    libaio-dev \
    liburing-dev \
    protobuf-compiler-grpc \
    libcpprest-dev \
    etcd-server \
    etcd-client \
    autotools-dev \
    automake \
    libtool \
    libz-dev \
    flex \
    build-essential

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
    --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev

WORKDIR /workspace
RUN git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git &&\
	cd etcd-cpp-apiv3 && mkdir build && cd build && \
	cmake .. && make -j$(nproc) && make install

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.86.0 \
    RUSTARCH=x86_64-unknown-linux-gnu

RUN wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    echo "a3339fb004c3d0bb9862ba0bce001861fe5cbde9c10d16591eb3f39ee6cd3e7f *rustup-init" | sha256sum -c - && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

WORKDIR /workspace/nixl
COPY . /workspace/nixl

# LD_LIBRARY_PATH is needed for auditwheel to find libcuda.so.1
# Set incorrectly to `compat/lib` in cuda-dl-base image
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/lib:$LD_LIBRARY_PATH

ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    # pybind11 pip install needed for ubuntu 22.04
    uv pip install --upgrade meson pybind11 patchelf

RUN rm -rf /usr/lib/ucx
RUN rm -rf /opt/hpcx/ucx

RUN cd /usr/local/src && \
     git clone https://github.com/openucx/ucx.git && \
     cd ucx && 			     \
     git checkout v1.19.x &&	     \
     ./autogen.sh && ./configure     \
         --enable-shared             \
         --disable-static            \
         --disable-doxygen-doc       \
         --enable-optimizations      \
         --enable-cma                \
         --enable-devel-headers      \
         --with-cuda=/usr/local/cuda \
         --with-verbs                \
         --with-dm                   \
         --with-gdrcopy=/usr/local   \
         --with-efa                  \
         --enable-mt &&              \
     make -j &&                      \
     make -j install-strip &&        \
     ldconfig

RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl && \
    cd build && \
    ninja && \
    ninja install

ENV NIXL_PREFIX=/usr/local/nixl
ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
    echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

RUN cd src/bindings/rust && cargo build --release --locked

# Create the wheel
ARG WHL_PYTHON_VERSIONS="3.12"
ARG WHL_PLATFORM="manylinux_2_39_x86_64"
RUN IFS=',' read -ra PYTHON_VERSIONS <<< "$WHL_PYTHON_VERSIONS" && \
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do \
        uv build --wheel --out-dir /tmp/dist --python $PYTHON_VERSION; \
    done
RUN uv pip install auditwheel && \
    uv run auditwheel repair /tmp/dist/nixl-*cp31*.whl --plat $WHL_PLATFORM --wheel-dir /workspace/nixl/dist

RUN uv pip install dist/nixl-*cp${DEFAULT_PYTHON_VERSION//./}*.whl

RUN uv pip install -r benchmark/ct_perftest/requirements.txt

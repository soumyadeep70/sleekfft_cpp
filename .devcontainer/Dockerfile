FROM mcr.microsoft.com/devcontainers/base:trixie

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        clangd \
        clang-format \
        clang-tidy \
        libclang-rt-dev \
        llvm \
        lldb \
        cmake \
        ninja-build \
        pkg-config \
        libclang-dev \
        libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

USER root
ENV DEBIAN_FRONTEND=dialog
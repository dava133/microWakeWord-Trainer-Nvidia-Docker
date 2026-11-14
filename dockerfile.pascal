# GTX 1060 / Pascal training image
# TensorFlow 2.10.1 GPU wheels target CUDA 11.2 + cuDNN 8.1 and support Pascal.
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MWW_PASCAL_IMAGE=1
ENV PYTHON=python3.10
# The trainer stages create their own PyTorch environments; pin those too.
ENV MWW_TORCH_CUDA=cu124
ENV MWW_TORCH_VERSION=2.5.0
ENV NWW_TORCH_CUDA=cu124
ENV NWW_TORCH_VERSION=2.5.0
ENV OWW_TORCH_CUDA=cu124
ENV OWW_TORCH_VERSION=2.5.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git wget unzip patch \
    ninja-build build-essential cmake pkg-config nano less libgomp1 \
    ffmpeg sox libsox-fmt-all libsndfile1 espeak-ng \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

EXPOSE 8789
WORKDIR /root/mww-scripts

COPY --chown=root:root --chmod=0755 .bashrc /root/
COPY --chown=root:root --chmod=0755 \
    train_wake_word \
    run.sh \
    trainer_server.py \
    openwakeword_stage.py \
    requirements.txt \
    /root/mww-scripts/
COPY --chown=root:root --chmod=0644 \
    tts_config.py \
    nanowakeword_stage.py \
    shared_feature_bank.py \
    training_summary.py \
    training_progress.py \
    wake_bundle.py \
    wake_word_library.py \
    /root/mww-scripts/
COPY --chown=root:root cli/ /root/mww-scripts/cli/
RUN chmod -R a+x /root/mww-scripts/cli
COPY --chown=root:root static/ /root/mww-scripts/static/
CMD ["/bin/bash", "-lc", "/root/mww-scripts/run.sh"]

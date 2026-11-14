# GTX 1060 (Pascal, compute capability 6.1) compatibility profile

The standard Docker image targets current GPUs and is not the right base for a GTX 1060's TensorFlow GPU stack. Build and run the Pascal-specific image instead:

```bash
docker build -f dockerfile.pascal -t microwakeword-trainer:pascal .
docker run --gpus all -p 8789:8789 -v /path/to/data:/data microwakeword-trainer:pascal
```

This profile selects:

- Python 3.10, because the TensorFlow 2.10 GPU wheel does not support Python 3.12.
- TensorFlow 2.10.1 with CUDA 11.2 and cuDNN 8.1, provided by the NVIDIA CUDA base image. TensorFlow 2.10's prebuilt Linux GPU binaries include Pascal support; later official TensorFlow Linux pip GPU binaries require compute capability 7.0 or newer.
- PyTorch and torchaudio 2.5.0 from the official `cu124` wheel index in the main environment and both external trainer environments. These wheels bundle their CUDA runtime; use a host NVIDIA driver new enough for CUDA 12.4.
- NumPy 1.23.5 and Keras 2.10.0 to meet TensorFlow 2.10's dependency constraints.
- CPU ONNX Runtime in the main trainer environment. Current feature extraction uses CPU ONNX; PyTorch remains GPU-enabled for the model optimization stages.

The image profile is explicit and is also selected automatically if the installer detects compute capability 6.1. It is not intended for Blackwell GPUs. Existing `/data/.venv` environments retain prior packages; delete that venv or use a fresh data directory when switching GPU compatibility profiles so the selected Python and framework stack are rebuilt.

Confirm passthrough and framework availability in the startup/environment test output. The host must have a compatible NVIDIA driver and NVIDIA Container Toolkit configured for `--gpus all`.

# Future work

Things to integrate into this TAS fork

## Node-based configuration

The GUI should use node-based configuration similar to [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) to configure model parameters, encoding settings etc.

## Encoding toolbox

Main reference: https://guide.encode.moe/index.html

- [getnative](https://github.com/Infiziert90/getnative)
- [descale](https://github.com/Irrational-Encoding-Wizardry/descale)
    - Also fix program to detect native 1080p (use outlier detection)
      See https://guide.encode.moe/encoding/descaling.html

## Models

### Repos

- https://github.com/NevermindNilas/TAS-Models-Host/releases/tag/main

### Shot Boundary Detection (Scene Detection)

Best

- https://github.com/UVA-Computer-Vision-Lab/OmniShotCut
- https://doi.org/10.48550/arXiv.2604.24762

Fallback

- https://github.com/Breakthrough/PySceneDetect

### Frame Deduplication

- https://github.com/routineLife1/MultiPassDedup

## Performance

General

- https://onnxruntime.ai/docs/performance/
- https://onnxruntime.ai/docs/execution-providers/
- https://github.com/microsoft/onnxruntime
- https://microsoft.github.io/Olive/
- https://onnxruntime.ai/docs/performance/device-tensor.html

Help

- https://onnxruntime.ai/docs/api/python/tutorial.html

### Intel

- https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#install

### AMD

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html#migraphx-execution-provider
- https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/install/install-torch-migraphx.html

### NVIDIA

- https://github.com/nvidia/Model-Optimizer

### Apple Silicon

- https://developer.apple.com/metal/pytorch/
- https://pytorch.org/blog/running-pytorch-models-on-apple-silicon-gpus-with-the-executorch-mlx-delegate/

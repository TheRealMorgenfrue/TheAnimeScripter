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

### Shot Boundary Detection (Scene Detection)

- https://github.com/UVA-Computer-Vision-Lab/OmniShotCut
- https://doi.org/10.48550/arXiv.2604.24762

## Performance

### Quantization

| $\textbf{Precision}^\dagger$ | Supported Architecture  | Efficiency |         Quality          |
| :--------------------------: | :---------------------: | :--------: | :----------------------: |
|             FP32             |           All           |     1x     |           Best           |
|             FP16             |  Turing GPUs or later   |   2x F32   | Slightly worse than FP32 |
|             FP8              |    Ada GPUs or later    |  4x FP32   | Slightly worse than FP16 |
|             FP4              | Blackwell GPUs or later |  8x FP32   | Slightly worse than FP8  |

$\dagger$ Generalized across variants

### Intel/AMD/NVIDIA

- https://onnxruntime.ai/docs/performance/
- https://onnxruntime.ai/docs/execution-providers/
- https://github.com/microsoft/onnxruntime
- https://microsoft.github.io/Olive/
- https://microsoft.github.io/Olive/privacy.html

### Apple Silicon

- https://developer.apple.com/metal/pytorch/
- https://pytorch.org/blog/running-pytorch-models-on-apple-silicon-gpus-with-the-executorch-mlx-delegate/

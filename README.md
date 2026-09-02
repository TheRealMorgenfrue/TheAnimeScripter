<div align="center">

<h1>
<img src="src/assets/logos/icon64x64.ico" alt="TAS Logo" width="32">
The Anime Scripter Redux
</h1>

#### _High-performance AI video enhancement toolkit for creators_

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F&labelColor=%23697689&countColor=%23ff8a65&style=flat-square&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FNevermindNilas%2FTheAnimeScripter%2F)
[![Release](https://img.shields.io/github/release/NevermindNilas/TheAnimeScripter.svg?style=flat-square&color=blue)](https://github.com/NevermindNilas/TheAnimeScripter/releases)
[![Downloads](https://img.shields.io/github/downloads/NevermindNilas/TheAnimeScripter/total.svg?style=flat-square&color=%2364ff82)](https://github.com/NevermindNilas/TheAnimeScripter/releases)
[![Last Commit](https://img.shields.io/github/last-commit/NevermindNilas/TheAnimeScripter.svg?style=flat-square)](https://github.com/NevermindNilas/TheAnimeScripter/commits)
[![Discord](https://img.shields.io/discord/1041502781808328704?style=flat-square&logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/hwGHXga8ck)
[![License](https://img.shields.io/github/license/NevermindNilas/TheAnimeScripter?style=flat-square&color=orange)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/NevermindNilas/TheAnimeScripter?style=flat-square&color=yellow)](https://github.com/NevermindNilas/TheAnimeScripter/stargazers)

</div>

# Overview

The Anime Scripter Redux is a complete rewrite of the popular video enhancement toolkit [The Anime Scripter](https://github.com/NevermindNilas/TheAnimeScripter).

## Goals

- Better performance
- Cross-platform (both software itself and model inference)
- Easier to use than the original
- Code maintainability (easier to deal in the code)
- Code modularity (easier to add future requested features)

## Non-goals

- Feature parity with [The Anime Scripter](https://github.com/NevermindNilas/TheAnimeScripter)

# Table of Contents

- [Key Features](#-key-features)
- [User Interfaces](#️-user-interfaces)
- [Installation Guide](#️-installation-guide)
- [Command Reference](#-command-reference)
- [Supported Models](#-supported-models)
- [Performance Benchmarks](#-performance-benchmarks)
- [Contributors](#-contributors)
- [Project Growth](#-project-growth)
- [Demo & Examples](#-demo--examples)

# Key Features

<table>
<tr>
<td width="50%">

## Video Enhancement

- **VFI** Create buttery-smooth animation with advanced frame interpolation
- **SR** Enhance resolution with AI-powered upscaling (2x)

</td>
<td width="50%">

## Quadrant Pending

</td>
</tr>
<tr>
<td>

## Workflow Optimization

- **Cross-platform:** Runs on Windows, Linux and MacOS
- **Model Chaining:** Combine multiple effects in a single processing pass
- **In-Memory Processing:** Efficient frame handling without redundant disk operations

</td>
<td>

## Model Flexibility

- **Multi-Backend Support:** CUDA, TensorRT, OpenVINO (WIP), ROCm (WIP), Metal (WIP), and NCNN (WIP) acceleration
- **Custom Model Support:** Import your own trained models

</td>
</tr>
</table>

# Getting Started

## System Requirements

The table below shows supported hardware combinations. For best performance, a GPU (or NPU/APU) is recommended.

|                                   Hardware                                    |      Operating System      | Supported |
| :---------------------------------------------------------------------------: | :------------------------: | :-------: |
|                   NVIDIA GPU <br> (GTX 10 series or later)                    |       Windows/Linux        |    ✅     |
|                   AMD GPU <br> (9000 & select 7000 Series)                    |       Windows/Linux        |    ✅     |
| AMD APU <br> (AI Max 300 Series, select AI 400 Series & select AI 300 Series) |       Windows/Linux        |    ✅     |
|                                 Intel GPU/NPU                                 |       Windows/Linux        |    ✅     |
|                                 Apple Silicon                                 | MacOS <br> (12.3 or later) |    ✅     |
|                       Intel CPU <br> (Haswell or later)                       |       Windows/Linux        |    ✅     |
|                         AMD CPU <br> (Zen1 or later)                          |       Windows/Linux        |    ✅     |

## Standalone Application (GUI)

> _Under active development - coming soon_

The native application will provide a dedicated environment optimized for batch processing and advanced customization options.

## Command-line Interface (CLI)

> _Under active development - coming soon_

# Development Setup

> [!NOTE]
> This project uses `uv` as package manager

0. Clone this repository

    ```bash
    git clone https://github.com/TheRealMorgenfrue/TheAnimeScripter
    ```

1. Install uv from https://docs.astral.sh/uv/getting-started/installation/
2. Figure out what GPU architecture you have:
    - RTX 50xx (Blackwell architecture, sm120)

    - RTX 40xx (Ada architecture, sm89)

    - RTX 30xx (Ampere architecture, sm86)

    - GTX 16xx/RTX 20xx (Turing architecture, sm75)

    - GTX 10xx (Pascal architecture, sm61)

3. Open a terminal in the `TheAnimeScripter` directory.
4. The installation differs slightly from here.
    - **CPU**:  
      Run `uv run main.py`
    - **GPU**:  
      Run `uv run --extra <GPU_ARCH> main.py`  
      Replace `<GPU_ARCH>` with your GPU architecture, e.g., `uv run --extra pascal main.py`

    It may take some time to install.  
    If a GUI displays, the installation was successful.

# Available Inputs

All available parameters for interacting with the CLI or directly with `main.py` can be found in the [Parameters](PARAMETERS.MD) guide.

# Available Models

## SR models

| Model                        | CUDA | TensorRT | DirectML | NCNN |
| ---------------------------- | :--: | :------: | :------: | :--: |
| Fallin Soft                  |  ✅  |    ✅    |    ✅    |  ❌  |
| Fallin Strong                |  ✅  |    ✅    |    ✅    |  ❌  |
| SRVGGNet (Compact)           |  ✅  |    ✅    |    ✅    |  ❌  |
| SRVGGNet (UltraCompact)      |  ✅  |    ✅    |    ✅    |  ❌  |
| SRVGGNet (SuperUltraCompact) |  ✅  |    ✅    |    ✅    |  ❌  |
| OpenProteus                  |  ✅  |    ✅    |    ✅    |  ❌  |
| AniScale 2                   |  ✅  |    ✅    |    ✅    |  ❌  |

## VFI models

| Version               | CUDA | TensorRT | DirectML | NCNN |
| --------------------- | :--: | :------: | :------: | :--: |
| Rife_Elexor (mod 4.7) |  ✅  |    ✅    |    ❌    |  ❌  |

# Project Contributors

## Model & Algorithm Contributors

| Contributor                                              | Type                  | Contribution                 | Repository                                                                 |
| -------------------------------------------------------- | --------------------- | ---------------------------- | -------------------------------------------------------------------------- |
| [HZWER](https://github.com/hzwer)                        | Interpolation (VFI)   | RIFE                         | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)                  |
| [Elexor](https://github.com/elexor)                      | Interpolation (VFI)   | Custom RIFE modifications    | [Modded Rife Experiment(s)](https://github.com/elexor)                     |
| [renarchi](https://github.com/renarchi)                  | Upscale (SR)          | Fallin' Soft & Strong models | [Fallin-Upscale](https://github.com/renarchi/Re-SISR)                      |
| [the-database](https://github.com/the-database)          | Upscale (SR)          | SRVGGNet model variants      | [2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai) |
| [Sirosky](https://github.com/Sirosky)                    | Upscale (SR)          | Open-Proteus & AniScale 2    | [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub)                      |
| [Breakthrough](https://github.com/Breakthrough)          | Scene Detection (SBD) | PySceneDetect algorithm      | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)             |
| [Wang et. al](https://doi.org/10.48550/arXiv.2604.24762) | Scene Detection (SBD) | OmniShotCut model            | [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut)      |

## Framework & Tool Contributors

| Contributor                         | Contribution                | Repository                                 |
| ----------------------------------- | --------------------------- | ------------------------------------------ |
| [FFmpeg Group](https://ffmpeg.org/) | Multimedia processing suite | [FFmpeg](https://github.com/FFmpeg/FFmpeg) |

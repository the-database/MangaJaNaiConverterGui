# <img src="logo.png" width="24"></img> MangaJaNaiConverterGui
[![Discord](https://img.shields.io/discord/1121653618173546546?label=Discord&logo=Discord&logoColor=white)](https://discord.gg/EeFfZUBvxj)

## Overview
This project provides a Windows GUI for upscaling manga images and archives with ONNX models, running on TensorRT, DirectML or (experimental) TensorRT-RTX through [animejanai-inference](https://github.com/the-database/animejanai-inference). It includes a set of models optimized for upscaling manga with Japanese and English text, but many other models are also supported. It also utilizes linear light downscaling to minimize halftone artifacts when downscaling. 

![image](https://github.com/the-database/MangaJaNaiConverterGui/assets/25811902/89095677-5b1f-46c9-9a1d-3d9df80cefe8)


## Instructions
Download and run `MangaJaNaiConverterGui-Setup-<version>.exe` from the [latest release](https://github.com/the-database/MangaJaNaiConverterGui/releases/latest). It installs for your user only, so no administrator rights are needed. Setup downloads the upscaling components for your GPU and the default models. Then select your input file or folder, choose your upscale settings, and click Upscale.

With **Auto Update** on (App Settings), updates download in the background and install when you close the app.

### Requirements
- An NVIDIA RTX GPU for the **TensorRT** backend (fastest), or any DirectX 12 GPU (NVIDIA, AMD, Intel) for the **DirectML** backend. Choose one in App Settings; **Auto** picks TensorRT when it can run.
- TensorRT builds an engine for each model and tile size the first time it is used, which takes a few minutes. Later upscales reuse it.
- **TensorRT-RTX** (experimental, NVIDIA RTX GPUs) builds its engines in about a second instead. Its output is equivalent to TensorRT's but not bit-identical, and it falls back to TensorRT when it can't start or can't build a model's engine, with a warning in the status bar. It is opt-in: install the **TensorRT-RTX** component in App Settings > Components (it needs the **TensorRT runtime** component too), then choose TensorRT-RTX as the backend. Auto never picks it.
- There's no CPU backend.

Keep your NVIDIA drivers up to date.

### Custom models
Put your own models in the models folder (App Settings > Models > Open Models Folder), then pick them in a chain. 2.0 runs ONNX models. Convert PyTorch models (`.pth`, `.safetensors`) to ONNX first: see [tools/onnx-export](tools/onnx-export).

### Upgrading from 1.x
Version 2.0 uses a new installer and doesn't update itself from 1.x. Install 2.0 with the setup above: it removes the 1.x install and its Python backend, and keeps your workflows and models folder. Workflows that use the default models switch to the ONNX versions automatically. Don't uninstall 1.x first; its uninstaller deletes the models folder, including any models you added.

### Command line
The upscaling backend, `inference\mangajanai_upscale.exe`, also runs on its own: `mangajanai_upscale.exe --help`.


## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models.

## Related Projects
- [MangaJaNai](https://github.com/the-database/mangajanai): Main repository for manga upscaling models.
- [VideoJaNai](https://github.com/the-database/VideoJaNai): Windows GUI for video upscaling with extremely fast performance.
- [traiNNer-redux](https://github.com/the-database/traiNNer-redux): Software for training upscaling models.

## Acknowledgments 
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling. MangaJaNaiConverterGui 1.x used a modified version of its backend for running upscaling models.

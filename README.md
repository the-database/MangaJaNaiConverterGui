# <img src="logo.png" width="24"></img> MangaJaNaiConverterGui
[![Discord](https://img.shields.io/discord/1121653618173546546?label=Discord&logo=Discord&logoColor=white)](https://discord.gg/EeFfZUBvxj)

## Overview
This project provides a Windows GUI for upscaling manga images and archives with PyTorch models. It includes a set of models optimized for upscaling manga with Japanese and English text, but many other models are also supported. It also utilizes linear light downscaling to minimize halftone artifacts when downscaling. 

![image](https://github.com/the-database/MangaJaNaiConverterGui/assets/25811902/89095677-5b1f-46c9-9a1d-3d9df80cefe8)


## Instructions
Download and run `MangaJaNaiConverterGui-Setup-<version>.exe` from the [latest release](https://github.com/the-database/MangaJaNaiConverterGui/releases/latest). It installs for your user only, so no administrator rights are needed. The first launch downloads the Python backend and models. Then select your input file or folder, choose your upscale settings, and click Upscale.

With **Auto Update** on (App Settings), updates download in the background and install when you close the app.

### Upgrading from 1.x
Version 2.0 uses a new installer and doesn't update itself from 1.x. Install 2.0 with the setup above: it removes the 1.x install and keeps your workflows, the Python backend and your models. Don't uninstall 1.x first; its uninstaller deletes the Python backend and the models folder, including any models you added.

### Important Note for NVIDIA Users

First, please ensure that you are running the latest NVIDIA drivers. To avoid major slowdowns while upscaling, open NVIDIA Control Panel and set the **CUDA - Sysmem Fallback Policy** setting to **Prefer No Sysmem Fallback**. 

![image](https://github.com/the-database/MangaJaNaiConverterGui/assets/25811902/3ad7392e-0de1-4eea-be59-a7b26935f08a)

### For Linux users
Check out [this README](MangaJaNaiConverterGui/backend/src/README.md).


## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models.

## Related Projects
- [MangaJaNai](https://github.com/the-database/mangajanai): Main repository for manga upscaling models.
- [VideoJaNai](https://github.com/the-database/VideoJaNai): Windows GUI for video upscaling with extremely fast performance.
- [traiNNer-redux](https://github.com/the-database/traiNNer-redux): Software for training upscaling models.

## Acknowledgments 
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling. This project uses a modified version of its backend for running upscaling models.

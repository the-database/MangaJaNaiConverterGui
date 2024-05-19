# <img src="logo.png" width="24"></img> MangaJaNaiConverterGui
[![Discord](https://img.shields.io/discord/1121653618173546546?label=Discord&logo=Discord&logoColor=white)](https://discord.gg/EeFfZUBvxj)

## Overview
This project provides a Windows GUI for upscaling manga images and archives with PyTorch models. It includes a set of models optimized for upscaling manga with Japanese and English text, but many other models are also supported. It also utilizes linear light downscaling to minimize halftone artifacts when downscaling. 

![image](https://github.com/the-database/MangaJaNaiConverterGui/assets/25811902/89095677-5b1f-46c9-9a1d-3d9df80cefe8)


## Instructions
Simply download  [MangaJaNaiConverterGui-win-Setup.exe](https://github.com/the-database/MangaJaNaiConverterGui/releases/latest/download/MangaJaNaiConverterGui-win-Setup.exe) (if you want to install the app) or [MangaJaNaiConverterGui-win-Portable.zip](https://github.com/the-database/MangaJaNaiConverterGui/releases/latest/download/MangaJaNaiConverterGui-win-Portable.zip) (if you want a portable version of the app that isn't installed) from the [latest release](https://github.com/the-database/MangaJaNaiConverterGui/releases). Select your input file or folder, choose your upscale settings, and click Upscale. 

### Important Note for NVIDIA Users

First, please ensure that you are running the latest NVIDIA drivers. To avoid major slowdowns while upscaling, open NVIDIA Control Panel and set the **CUDA - Sysmem Fallback Policy** setting to **Prefer No Sysmem Fallback**. 

![image](https://github.com/the-database/MangaJaNaiConverterGui/assets/25811902/3ad7392e-0de1-4eea-be59-a7b26935f08a)



## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models.

## Related Projects
- [MangaJaNai](https://github.com/the-database/mangajanai): Main repository for manga upscaling models
- [AnimeJaNaiConverterGui](https://github.com/the-database/AnimeJaNaiConverterGui): Windows GUI for video upscaling with extremely fast performance

## Acknowledgments 
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling. This project uses its backend for running upscaling models.
- [7-zip](https://www.7-zip.org/) and [SevenZipExtractor](https://github.com/adoconnection/SevenZipExtractor): File archiver and C# wrapper for 7z.dll.

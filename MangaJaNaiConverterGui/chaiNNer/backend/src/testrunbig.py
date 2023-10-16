import asyncio
import functools
import gc
import importlib
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from json import dumps as stringify
from typing import Dict, List, Optional, Tuple, TypedDict, Union
from nodes.utils.utils import get_h_w_c
from pathlib import Path

import psutil
from sanic import Sanic
from sanic.log import access_logger, logger
from sanic.request import Request
from sanic.response import json
from sanic_cors import CORS
import numpy as np
from PIL import Image
import io
import cv2

import api
from base_types import NodeId
from chain.cache import OutputCache
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from custom_types import UpdateProgressFn
from dependencies.store import DependencyInfo, install_dependencies, installed_packages
from events import EventQueue, ExecutionErrorData
from gpu import get_nvidia_helper
from nodes.group import Group
from nodes.impl.image_utils import cv_save_image, to_uint8, to_uint16
from nodes.utils.exec_options import (
    ExecutionOptions,
    JsonExecutionOptions,
    set_execution_options,
)
from process import (
    Executor,
    NodeExecutionError,
    Output,
    compute_broadcast,
    run_node,
    timed_supplier,
)
from progress_controller import Aborted
from response import (
    alreadyRunningResponse,
    errorResponse,
    noExecutorResponse,
    successResponse,
)
from server_config import ServerConfig
from system import is_arm_mac

from packages.chaiNNer_standard.image.io.load_image import load_image_node
from packages.chaiNNer_standard.image_adjustment.adjustments.stretch_contrast import stretch_contrast_node, StretchMode
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import upscale_image_node
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    NO_TILING,
    TileSize,
    estimate_tile_size,
    parse_tile_size_input,
)
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool

from PIL import Image, ImageOps
import numpy as np

def find_peaks(a):
  x = np.array(a)
  max = np.max(x)
  length = len(a)
  ret = []
  for i in range(length):
      ispeak = True
      if i-1 > 0:
          ispeak &= (x[i] > 1.8 * x[i-1])
      if i+1 < length:
          ispeak &= (x[i] > 1.8 * x[i+1])

      ispeak &= (x[i] > 0.1 * max)
      if ispeak:
          ret.append(i)
  return ret

def enhance_contrast(image):
    # print('1', image[199][501], np.min(image), np.max(image))
    image_p = Image.fromarray(image).convert("L")

    # Calculate the histogram
    hist = image_p.histogram()
    print(hist)
    print('peak',find_peaks(hist[:100]))

    # Find the global maximum peak in the range 0-30 for the black level
    new_black_level = 1
    global_max_black = hist[1]

    for i in range(2, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i
        # elif hist[i] < global_max_black:
        #     break
    print('1',new_black_level)
    # Continue searching at 31 and later for the black level
    continuous_count = 0
    for i in range(31, 256):
        if hist[i] > global_max_black:
            continuous_count = 0
            global_max_black = hist[i]
            new_black_level = i
        elif hist[i] < global_max_black:
            continuous_count += 1
            if continuous_count > 1:
                break

    # Find the global maximum peak in the range 255-225 for the white level
    new_white_level = 255
    global_max_white = hist[255]

    for i in range(254, 224, -1):
        if hist[i] > global_max_white:
            global_max_white = hist[i]
            new_white_level = i
        # elif hist[i] < global_max_white:
        #     break

    # Continue searching at 224 and below for the white level
    continuous_count = 0
    for i in range(223, -1, -1):
        if hist[i] > global_max_white:
            continuous_count = 0
            global_max_white = hist[i]
            new_white_level = i
        elif hist[i] < global_max_white:
            continuous_count += 1
            if continuous_count > 1:
                break

    # print("NEW BLACK LEVEL =", new_black_level)
    # print("NEW WHITE LEVEL =", new_white_level)

    # Apply level adjustment
    # min_pixel_value = np.min(image)
    # max_pixel_value = np.max(image)
    # adjusted_image = ImageOps.level(image, (min_pixel_value, max_pixel_value), (new_black_level, new_white_level))


    # print("np.max =", new_black_level)
    # Create a NumPy array from the image
    image_array = np.array(image_p).astype('float32')
    # print('2', image_array[199][501], np.min(image), np.max(image))
    # new_black_level = np.max(image_array)
    # print(image_array)
    # Apply level adjustment
    # min_pixel_value = np.min(image_array)
    # max_pixel_value = np.max(image_array)

    # Normalize pixel values to the new levels
    # print(image_array)
    image_array = np.maximum(image_array - new_black_level, 0) / (new_white_level - new_black_level)
    image_array = np.clip(image_array, 0, 1)
    # print('3', image_array[199][501], np.min(image), np.max(image))
    # print(image_array)
    # print(np.any(image_array > 1))

    # Ensure the pixel values are in the valid range [0, 255]


    # print(image_array)
    # Create a new Pillow image from the adjusted NumPy array
    # return stretch_contrast_node(image, StretchMode.MANUAL, True, 0, 50, 100)
    return new_black_level, new_white_level

# # Example usage:
# input_image_path = 'input_image.jpg'
# output_image_path = 'enhanced_image.jpg'
# input_image = Image.open(input_image_path).convert('L')
# enhanced_image = enhance_contrast(input_image)
# enhanced_image.save(output_image_path)
# print("Enhanced image saved as", output_image_path)






searchdir = r"D:\file\chainner\4xMangaJaNai\original 4x"
images = []

model, dirname, basename = load_model_node(r"\\WEJJ-II\traiNNer-redux\experiments\4x_MangaJaNai_V1RC24_OmniSR\models\net_g_40000.pth")
# model, dirname, basename = load_model_node(r"C:\mpv-upscale-2x_animejanai\vapoursynth64\plugins\models\animejanai\4x_MangaJaNai_V1_RC24_OmniSR_40k.onnx")
# model, dirname, basename = load_model_node(r"D:\file\VSGAN-tensorrt-docker\models\1x_AnimeUndeint_Compact_130k_net_g.pth")

# Record the start time
# start_time = time.time()


# for filename in os.listdir(searchdir):
#   img_a, img_dir_a, basename_a = load_image_node(os.path.join(r"D:\file\chainner\4xMangaJaNai\original 4x", filename))
#   images.append(img_a)

# for img_a in images:
#   upscaled_image = upscale_image_node(img_a, model, NO_TILING, False)
# # def worker(f):
# #     try:
# #         upscale_image_node(f, model, ESTIMATE, False)
# #     except Exception as e:
# #         print(e)
# # pool_size = 4
# # with Pool(pool_size) as p:
# #     r = list(tqdm(p.imap(worker, images), total=len(images)))

# # Record the end time
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time

# # Print the elapsed time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")













import zipfile

def _read_pil(im) -> np.ndarray | None:
    img = np.array(im)
    _, _, c = get_h_w_c(img)
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img

# Define the paths for the input and output zip files
input_path = r"\\wejj-ii\traiNNer-redux\datasets\train\4x-mangajanai new base\lr degraded base magick 3 cropped\12.Ans.T08-3006"
# input_path = r"D:\file\traiNNer-redux\datasets\train\4x-mangajanai workspace\manga raw pdfs hr curated inverted 4800"
# input_path = r"C:\Users\jsoos\Documents\Calibre Library\Unknown\dl3pahxr (1096)\dl3pahxr - Unknown\OPS\images"

count = 0
failures = []

for root, _, files in tqdm(os.walk(input_path)):
    for file_name in files:


        # if '280' in file_name:
        if True:
            # Open the image using Pillow (PIL)
            with Image.open(os.path.join(root, file_name)) as img:
                image = _read_pil(img)
                new_black_level, new_white_level = enhance_contrast(image)
                count += 1
                if new_black_level != 0 or new_white_level != 255:
                    fail = (file_name, new_black_level, new_white_level)
                    print(fail)
                    failures.append(fail)
                # print(file_name, new_black_level, new_white_level)
                # print(image[199][501])
                # image = upscale_image_node(image, model, NO_TILING, False)
                # image = to_uint8(image, normalized=True)

                # Convert the resized image back to bytes
                # basename = Path(file_name).stem
                # output_file_path = os.path.join(output_path, f'{basename}_autolevels.png')
                # Image.fromarray(image).save(output_file_path, format="PNG")

print(f'{len(failures)}/{count} failed')

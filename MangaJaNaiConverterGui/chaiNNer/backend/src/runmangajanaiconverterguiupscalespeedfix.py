import sys
import asyncio
import functools
import gc
import importlib
import logging
import os
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
import argparse
import zipfile
import time
from multiprocessing import Queue, Process, Manager

def enhance_contrast(image):
    # print('1', image[199][501], np.min(image), np.max(image))
    image_p = Image.fromarray(image).convert("L")

    # Calculate the histogram
    hist = image_p.histogram()
    # print(hist)

    # Find the global maximum peak in the range 0-30 for the black level
    new_black_level = 0
    global_max_black = hist[0]

    for i in range(1, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i
        # elif hist[i] < global_max_black:
        #     break

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

    print(f"Auto adjusted levels: new black level = {new_black_level}; new white level = {new_white_level}")


    image_array = np.array(image_p).astype('float32')
    # image_array = np.maximum(image_array - new_black_level, 0) / (new_white_level - new_black_level)
    # image_array = np.clip(image_array, 0, 1)

    return image_array

def _read_pil(im) -> np.ndarray | None:
    img = np.array(im)
    _, _, c = get_h_w_c(img)
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def load_image(image_data):
    # print(f"load_image")
    with Image.open(io.BytesIO(image_data)) as img:
        return _read_pil(img)
    return None

def preprocess_image(image):
    # print(f"preprocess_image")
    return enhance_contrast(image)

def ai_upscale_image(image):
    print(f"ai_upscale_image")
    # if grayscale_model is not None:
        # TODO estimate vs no_tiling benchmark
    return upscale_image_node(image, grayscale_model, NO_TILING, False)  # TODO color vs grayscale model
    # return image
    # return (image, file_name)

def postprocess_image(image):
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)

def save_image_zip(image, file_name, output_zip, image_format, lossy_compression_quality, use_lossless_compression):
    print(f"save image to zip: {file_name}")


    # Convert the resized image back to bytes
    output_buffer = io.BytesIO()
    Image.fromarray(image).save(output_buffer, format="PNG")#TODO
    upscaled_image_data = output_buffer.getvalue()

    # Add the resized image to the output zip
    output_zip.writestr(file_name, upscaled_image_data)


def save_image(image, output_file_path, image_format, lossy_compression_quality, use_lossless_compression):
    print(f"save image: {output_file_path}")
    Image.fromarray(image).save(output_file_path, format=image_format, quality=lossy_compression_quality, lossless=use_lossless_compression)


def preprocess_worker_zip(preprocess_queue, upscale_queue, input_zip_path, auto_adjust_levels):
    """
    given a zip path, read images out of the zip, apply auto levels, add the image to upscale queue
    """
    print("preprocess_worker_zip entering")

    with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
        # Create a new zip file in write mode for the resized images
        #with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        # Iterate through the files in the input zip
        for file_name in input_zip.namelist():
            # Open the file inside the input zip
            with input_zip.open(file_name) as file_in_zip:
                # Read the image data
                # load_queue.put((file_in_zip.read(), file_name))

                image_data = file_in_zip.read()

                try:
                    with Image.open(io.BytesIO(image_data)) as img:
                        image = _read_pil(img)
                        # if auto_adjust_levels:
                        #     image = enhance_contrast(image)
                        # else:
                        image = image.astype('float32')
                        print(f"add to upscale_queue: {file_name}")
                        upscale_queue.put((image, file_name))

                except:
                    # skip non-images
                    # TODO copy non-images. copy queue?
                    # print(f"could not read as image, copying file to zip instead of upscaling: {file_name}")
                    # upscale_queue.put((image_data, file_name))
                    pass
    upscale_queue.put(SENTINEL)

    print("preprocess_worker_zip exiting")


def preprocess_worker_folder(preprocess_queue, upscale_queue, input_folder_path, output_folder_path, upscale_images, upscale_archives,
overwrite_existing_files, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):
    """
    given a folder path, recursively iterate the folder
    """
    print("preprocess_worker_folder entering")
    for root, dirs, files in os.walk(input_folder_path):
        for filename in files:
            # for output file, create dirs if necessary, or skip if file exists and overwrite not enabled
            filename_rel = os.path.relpath(os.path.join(root, filename), input_folder_path)
            output_file_path = Path(os.path.join(output_folder_path, filename_rel)).with_suffix(f'.{image_format}')
            if not overwrite_existing_files and os.path.isfile(output_file_path):
                print(f"file exists, skip: {output_file_path}")
                continue

            if filename.lower().endswith(IMAGE_EXTENSIONS): # TODO if image
                if upscale_images:
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with Image.open(os.path.join(root, filename)) as img:
                        image = _read_pil(img)
                        if auto_adjust_levels:
                            image = enhance_contrast(image)
                        upscale_queue.put((image, filename_rel))
            elif filename.lower().endswith(('.zip', '.cbz')):  # TODO if archive
                if upscale_archives:
                    os.makedirs(os.path.dirname(os.path.join(output_folder_path, filename_rel)), exist_ok=True)
                    upscale_zip_file(os.path.join(root, filename), os.path.join(output_folder_path, filename_rel), auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression) # TODO custom output extension
    upscale_queue.put(SENTINEL)
    print("preprocess_worker_folder exiting")


def preprocess_worker_image(preprocess_queue, upscale_queue, input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels):
    """
    given an image path, apply auto levels and add to upscale queue
    """

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}")
            return

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with Image.open(input_image_path) as img:
            image = _read_pil(img)
            if auto_adjust_levels:
                image = enhance_contrast(image)
            upscale_queue.put((image, None))
    upscale_queue.put(SENTINEL)


def upscale_worker(upscale_queue, postprocess_queue):
    """
    wait for upscale queue, for each queue entry, upscale image and add result to postprocess queue
    """
    print("upscale_worker entering")
    while True:
        image, file_name = upscale_queue.get()
        if image is None:
            break

        # if is_image:
        image = ai_upscale_image(image)
        postprocess_queue.put((image, file_name))
    postprocess_queue.put(SENTINEL)
    print("upscale_worker exiting")

def postprocess_worker_zip(postprocess_queue, output_zip_path, image_format, lossy_compression_quality, use_lossless_compression):
    """
    wait for postprocess queue, for each queue entry, save the image to the zip file
    """
    print("postprocess_worker_zip entering")
    with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        while True:
            image, file_name = postprocess_queue.get()
            if image is None:
                break
            # if is_image:
            image = postprocess_image(image)
            save_image_zip(image, file_name, output_zip, image_format, lossy_compression_quality, use_lossless_compression)
            # else:  # copy file
                # output_zip.writestr(file_name, image)

    print("postprocess_worker_zip exiting")


def postprocess_worker_folder(postprocess_queue, output_folder, image_format, lossy_compression_quality, use_lossless_compression):
    """
    wait for postprocess queue, for each queue entry, save the image to the output folder
    """
    print("postprocess_worker_folder entering")
    while True:
        image, file_name = postprocess_queue.get()
        if image is None:
            break
        image = postprocess_image(image)
        save_image(image, os.path.join(output_folder, str(Path(file_name).with_suffix(f'.{image_format}'))), image_format, lossy_compression_quality, use_lossless_compression)

    print("postprocess_worker_folder exiting")


def postprocess_worker_image(postprocess_queue, output_file_path, image_format, lossy_compression_quality, use_lossless_compression):
    """
    wait for postprocess queue, for each queue entry, save the image to the output file path
    """
    while True:
        image, _ = postprocess_queue.get()
        if image is None:
            break
        image = postprocess_image(image)
        save_image(image, output_file_path, image_format, lossy_compression_quality, use_lossless_compression)


def upscale_zip_file(input_zip_path, output_zip_path, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):
    # TODO accept multiple paths to reuse simple queues?

    preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Process(target=preprocess_worker_zip, args=(preprocess_queue, upscale_queue, input_zip_path, auto_adjust_levels))
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
    upscale_process.start()

    # start postprocess zip process
    postprocess_process = Process(target=postprocess_worker_zip, args=(postprocess_queue, output_zip_path, image_format, lossy_compression_quality, use_lossless_compression))
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_image_file(input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):
    preprocess_queue = Queue()
    upscale_queue = Queue()
    postprocess_queue = Queue()

    preprocess_processes = []
    for _ in range(PREPROCESS_MAX_PROCESSES):  # You can adjust the number of preprocess workers as needed
        p = Process(target=preprocess_worker_image, args=(preprocess_queue, upscale_queue, input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels))
        preprocess_processes.append(p)
        p.start()

    upscale_processes = []
    for _ in range(UPSCALE_MAX_PROCESSES):  # Limit to 1 task in parallel
        p = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
        upscale_processes.append(p)
        p.start()

    postprocess_processes = []
    for _ in range(POSTPROCESS_MAX_PROCESSES):  # You can adjust the number of preprocess workers as needed
        p = Process(target=postprocess_worker_image, args=(postprocess_queue, output_image_path, image_format, lossy_compression_quality, use_lossless_compression))
        postprocess_processes.append(p)
        p.start()




    print("0.5!!!!!!!!!!!!!!!!????????????????????????????????????????????????????????????????")
    for p in preprocess_processes:
        p.join()
    print("1?????????????????????????????????????")
    for p in upscale_processes:
        p.join()
    print("2???????????????????????????????????")
    for p in postprocess_processes:
        p.join()
    print("3??????????????????????????????????????")


def upscale_file(input_file, output_file, overwrite_existing_files,
auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):

    if not overwrite_existing_files and os.path.isfile(output_file):
        print(f"file exists, skip: {output_file}")
        return

    if input_file.lower().endswith('.zip'):  # TODO if archive
        upscale_zip_file(input_file, output_file, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression)
    elif input_file.lower().endswith(IMAGE_EXTENSIONS): # TODO if image
        upscale_image_file(input_file, output_file, overwrite_existing_files, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression)


def upscale_folder(input_folder, output_folder, upscale_images, upscale_archives, overwrite_existing_files,
auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):
    print("upscale_folder: entering")

    preprocess_queue = Queue()
    upscale_queue = Queue()
    postprocess_queue = Queue()

    # start preprocess folder process
    preprocess_process = Process(target=preprocess_worker_folder, args=(preprocess_queue, upscale_queue, input_folder, output_folder,
        upscale_images, upscale_archives, overwrite_existing_files, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression))
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
    upscale_process.start()

    # start postprocess folder process
    postprocess_process = Process(target=postprocess_worker_folder, args=(postprocess_queue, output_folder,
        image_format, lossy_compression_quality, use_lossless_compression))
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


# parser = argparse.ArgumentParser()

# parser.add_argument('--input-file', required=False)
# parser.add_argument('--output-file', required=False)
# parser.add_argument('--input-folder', required=False)
# parser.add_argument('--output-folder', required=False)
# parser.add_argument('--upscale-archives', action=argparse.BooleanOptionalAction)
# parser.add_argument('--upscale-images', action=argparse.BooleanOptionalAction)
# parser.add_argument('--overwrite-existing-files', action=argparse.BooleanOptionalAction)
# parser.add_argument('--auto-adjust-levels', action=argparse.BooleanOptionalAction)
# parser.add_argument('--image-format')
# parser.add_argument('--lossy-compression-quality')
# parser.add_argument('--use-lossless-compression', action=argparse.BooleanOptionalAction)
# parser.add_argument('--grayscale-model-path', required=False)
# parser.add_argument('--color-model-path', required=False)

# args = parser.parse_args()
# print(args)


SENTINEL = (None, None)
#IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
#ARCHIVE_EXTENSIONS = ('.zip', '.cbz')
# color_model = None
# grayscale_model = None

# if args.color_model_path:
    # color_model, dirname, basename = load_model_node(args.color_model_path)

# if args.grayscale_model_path:
grayscale_model, dirname, basename = load_model_node(r"\\WEJJ-II\traiNNer-redux\experiments\4x_MangaJaNai_V1RC29_OmniSR\models\net_g_40000.pth")

if __name__ == '__main__':
    #gc.disable() #TODO!!!!!!!!!!!!
    # Record the start time
    start_time = time.time()

    # if args.input_folder:
        # upscale_folder(args.input_folder, args.output_folder, args.upscale_images, args.upscale_archives, args.overwrite_existing_files, args.auto_adjust_levels, args.image_format, args.lossy_compression_quality, bool(args.use_lossless_compression))
    # elif args.input_file:
    upscale_zip_file(
        r"C:\Users\jsoos\Documents\Calibre Library\Unknown\dl3pahxr (1096)\dl3pahxr - Unknown\OPS\c127.zip",
        r"C:\Users\jsoos\Documents\Calibre Library\Unknown\dl3pahxr (1096)\dl3pahxr - Unknown\OPS\c127benchsystempy.zip",
        # True,
        False,
        "png",
        "80",
        False,
    )
    # upscale_zip_file(
    #     input_file,
    #     output_file,
    #     auto_adjust_levels,
    #     image_format,
    #     lossy_compression_quality,
    #     use_lossless_compression
    # )


    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

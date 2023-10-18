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
from ctypes import windll

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


def get_system_codepage():
    return windll.kernel32.GetConsoleOutputCP()


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

    print(f"Auto adjusted levels: new black level = {new_black_level}; new white level = {new_white_level}", flush=True)

    image_array = np.array(image_p).astype('float32')
    image_array = np.maximum(image_array - new_black_level, 0) / (new_white_level - new_black_level)
    image_array = np.clip(image_array, 0, 1)

    return image_array


def _read_cv(img_stream):
    # if get_ext(path) not in get_opencv_formats():
    #     # not supported
    #     return None

    # img = None
    # try:
    #     img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # except Exception as cv_err:
    #     print(f"Error loading image, trying with imdecode: {cv_err}")

    # if img is None:
    #     try:
    #         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #     except Exception as e:
    #         raise RuntimeError(
    #             f'Error reading image image from path "{path}". Image may be corrupt.'
    #         ) from e

    # if img is None:  # type: ignore
    #     raise RuntimeError(
    #         f'Error reading image image from path "{path}". Image may be corrupt.'
    #     )

    return cv2.imdecode(np.frombuffer(img_stream.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)



def _read_cv_from_path(path):
    img = None
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as cv_err:
        logger.warning(f"Error loading image, trying with imdecode: {cv_err}")

    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise RuntimeError(
                f'Error reading image image from path "{path}". Image may be corrupt.'
            ) from e

    if img is None:  # type: ignore
        raise RuntimeError(
            f'Error reading image image from path "{path}". Image may be corrupt.'
        )

    return img


def _read_pil(im) -> np.ndarray | None:
    img = np.array(im)
    _, _, c = get_h_w_c(img)
    print(f"_read_pil c={c}")
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def ai_upscale_image(image):
    # print(f"ai_upscale_image")
    if grayscale_model is not None:
        # TODO estimate vs no_tiling benchmark
        return upscale_image_node(image, grayscale_model, ESTIMATE, False)  # TODO color vs grayscale model
    return image

def postprocess_image(image):
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)

def save_image_zip(image, file_name, output_zip, image_format, lossy_compression_quality, use_lossless_compression):
    # sys.stdout.reconfigure(encoding='utf-8')  # TODO remove
    print(f"save image to zip: {file_name}", flush=True)

    params = []
    if image_format == 'jpg':
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(lossy_compression_quality),
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            1, # jpeg_progressive
        ]
    elif image_format == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 101 if use_lossless_compression else int(lossy_compression_quality)]
    else:
        params = []

    # the bit depth depends on the image format and settings
    precision = "u8"

    if precision == "u8":
        image = to_uint8(image, normalized=True)
    elif precision == "u16":
        image = to_uint16(image, normalized=True)
    elif precision == "f32":
        # chainner images are always f32
        pass

    # cv_save_image(output_file_path, image, params)

    # Convert the resized image back to bytes
    _, buf_img = cv2.imencode(f".{image_format}", image, params)
    output_buffer = io.BytesIO(buf_img)
    upscaled_image_data = output_buffer.getvalue()

    # Add the resized image to the output zip
    output_zip.writestr(file_name, upscaled_image_data)


def save_image(image, output_file_path, image_format, lossy_compression_quality, use_lossless_compression):
    print(f"save image: {output_file_path}")

    # channels = get_h_w_c(image)[2]
    # print(f"channels = {channels}")
    # if channels == 1:
    #     # PIL supports grayscale images just fine, so we don't need to do any conversion
    #     pass
    # elif channels == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # elif channels == 4:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    # else:
    #     raise RuntimeError(
    #         f"Unsupported number of channels. Saving .{image_format.extension} images is only supported for "
    #         f"grayscale, RGB, and RGBA images."
    #     )

    # Image.fromarray(image).save(output_file_path, format=image_format, quality=lossy_compression_quality, lossless=use_lossless_compression)


    params = []
    if image_format == 'jpg':
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(lossy_compression_quality),
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            1, # jpeg_progressive
        ]
    elif image_format == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 101 if use_lossless_compression else int(lossy_compression_quality)]
    else:
        params = []

    # the bit depth depends on the image format and settings
    precision = "u8"

    if precision == "u8":
        image = to_uint8(image, normalized=True)
    elif precision == "u16":
        image = to_uint16(image, normalized=True)
    elif precision == "f32":
        # chainner images are always f32
        pass

    cv_save_image(output_file_path, image, params)



def preprocess_worker_zip(upscale_queue, input_zip_path, auto_adjust_levels):
    """
    given a zip path, read images out of the zip, apply auto levels, add the image to upscale queue
    """
    print(f"preprocess_worker_zip entering aal={auto_adjust_levels}")

    with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
        # Create a new zip file in write mode for the resized images
        #with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        # Iterate through the files in the input zip
        for filename in input_zip.namelist():
            decoded_filename = filename
            try:
                decoded_filename = decoded_filename.encode('cp437').decode(f'cp{system_codepage}')
            except:
                pass

            # Open the file inside the input zip
            with input_zip.open(filename) as file_in_zip:
                # Read the image data
                # load_queue.put((file_in_zip.read(), filename))

                image_data = file_in_zip.read()

                try:
                    # with Image.open(io.BytesIO(image_data)) as img:
                    image_bytes = io.BytesIO(image_data)
                    # image = _read_pil(img)
                    image = _read_cv(image_bytes)
                    if auto_adjust_levels:
                        image = enhance_contrast(image)
                    else:
                        # TODO ???
                        # image = image.astype('float32')
                        # print("?!?!?!?!?")
                        # image = np.array(Image.fromarray(image).convert("L")).astype('float32')
                        pass
                    upscale_queue.put((image, decoded_filename, True))
                except:
                    print(f"could not read as image, copying file to zip instead of upscaling: {decoded_filename}")
                    upscale_queue.put((image_data, decoded_filename, False))
                    pass
    upscale_queue.put(SENTINEL)

    print("preprocess_worker_zip exiting")


def preprocess_worker_folder(upscale_queue, input_folder_path, output_folder_path, upscale_images, upscale_archives,
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
                    image = _read_cv_from_path(os.path.join(root, filename))
                    # with Image.open(os.path.join(root, filename)) as img:
                        # image = _read_pil(img)
                    if auto_adjust_levels:
                        image = enhance_contrast(image)
                    else:
                        # image = np.array(Image.fromarray(image).convert("L")).astype('float32')  # TODO ???
                        pass
                    upscale_queue.put((image, filename_rel, True))
            elif filename.lower().endswith(ZIP_EXTENSIONS):  # TODO if archive
                if upscale_archives:
                    os.makedirs(os.path.dirname(os.path.join(output_folder_path, filename_rel)), exist_ok=True)
                    upscale_zip_file(os.path.join(root, filename), os.path.join(output_folder_path, filename_rel), auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression) # TODO custom output extension
    upscale_queue.put(SENTINEL)
    print("preprocess_worker_folder exiting")


def preprocess_worker_image(upscale_queue, input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels):
    """
    given an image path, apply auto levels and add to upscale queue
    """

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}")
            return

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # with Image.open(input_image_path) as img:
        image = _read_cv_from_path(input_image_path)
        # image = _read_pil(img)
        if auto_adjust_levels:
            image = enhance_contrast(image)
        else:
            print("why?")
            # image_p = Image.fromarray(image).convert("L")
            # image_array = np.array(image_p).astype('float32')
            # image_array = np.maximum(image_array - 0, 0) / (255 - 0)
            # image_array = np.clip(image_array, 0, 1)
        upscale_queue.put((image, None, True))
    upscale_queue.put(SENTINEL)


def upscale_worker(upscale_queue, postprocess_queue):
    """
    wait for upscale queue, for each queue entry, upscale image and add result to postprocess queue
    """
    print("upscale_worker entering")
    while True:
        image, file_name, is_image = upscale_queue.get()
        if image is None:
            break

        if is_image:
            image = ai_upscale_image(image)
        postprocess_queue.put((image, file_name, is_image))
    postprocess_queue.put(SENTINEL)
    print("upscale_worker exiting")

def postprocess_worker_zip(postprocess_queue, output_zip_path, image_format, lossy_compression_quality, use_lossless_compression):
    """
    wait for postprocess queue, for each queue entry, save the image to the zip file
    """
    print("postprocess_worker_zip entering")
    with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        while True:
            image, file_name, is_image = postprocess_queue.get()
            if image is None:
                break
            if is_image:
                # image = postprocess_image(image)
                save_image_zip(image, str(Path(file_name).with_suffix(f'.{image_format}')), output_zip, image_format, lossy_compression_quality, use_lossless_compression)
            else:  # copy file
                output_zip.writestr(file_name, image)

    print("postprocess_worker_zip exiting")


def postprocess_worker_folder(postprocess_queue, output_folder, image_format, lossy_compression_quality, use_lossless_compression):
    """
    wait for postprocess queue, for each queue entry, save the image to the output folder
    """
    print("postprocess_worker_folder entering")
    while True:
        image, file_name, _ = postprocess_queue.get()
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
        image, _, _ = postprocess_queue.get()
        if image is None:
            break
        # image = postprocess_image(image)
        save_image(image, output_file_path, image_format, lossy_compression_quality, use_lossless_compression)


def upscale_zip_file(input_zip_path, output_zip_path, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):
    # TODO accept multiple paths to reuse simple queues?
    # print("hello?",flush=True)
    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Process(target=preprocess_worker_zip, args=(upscale_queue, input_zip_path, auto_adjust_levels))
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
    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Process(target=preprocess_worker_image, args=(upscale_queue, input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels))
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
    upscale_process.start()

    # start postprocess zip process
    postprocess_process = Process(target=postprocess_worker_image, args=(postprocess_queue, output_image_path, image_format, lossy_compression_quality, use_lossless_compression))
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_file(input_file, output_file, overwrite_existing_files,
auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):

    if not overwrite_existing_files and os.path.isfile(output_file):
        print(f"file exists, skip: {output_file}")
        return

    if input_file.lower().endswith(ZIP_EXTENSIONS):  # TODO if archive
        upscale_zip_file(input_file, output_file, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression)
    elif input_file.lower().endswith(IMAGE_EXTENSIONS): # TODO if image
        upscale_image_file(input_file, output_file, overwrite_existing_files, auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression)


def upscale_folder(input_folder, output_folder, upscale_images, upscale_archives, overwrite_existing_files,
auto_adjust_levels, image_format, lossy_compression_quality, use_lossless_compression):
    print("upscale_folder: entering")

    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess folder process
    preprocess_process = Process(target=preprocess_worker_folder, args=(upscale_queue, input_folder, output_folder,
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

sys.stdout.reconfigure(encoding='utf-8')
parser = argparse.ArgumentParser()

parser.add_argument('--input-file', required=False)
parser.add_argument('--output-file', required=False)
parser.add_argument('--input-folder', required=False)
parser.add_argument('--output-folder', required=False)
parser.add_argument('--upscale-archives', action=argparse.BooleanOptionalAction)
parser.add_argument('--upscale-images', action=argparse.BooleanOptionalAction)
parser.add_argument('--overwrite-existing-files', action=argparse.BooleanOptionalAction)
parser.add_argument('--auto-adjust-levels', action=argparse.BooleanOptionalAction)
parser.add_argument('--image-format')
parser.add_argument('--lossy-compression-quality')
parser.add_argument('--use-lossless-compression', action=argparse.BooleanOptionalAction)
parser.add_argument('--grayscale-model-path', required=False)
parser.add_argument('--color-model-path', required=False)

args = parser.parse_args()
print(args)
print(args.input_file)


SENTINEL = (None, None, None)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
ZIP_EXTENSIONS = ('.zip', '.cbz')
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS # TODO .rar .cbr .7z
color_model = None
grayscale_model = None
system_codepage = get_system_codepage()

if args.color_model_path:
    color_model, dirname, basename = load_model_node(args.color_model_path)

if args.grayscale_model_path:
    grayscale_model, dirname, basename = load_model_node(args.grayscale_model_path)




if __name__ == '__main__':
    #gc.disable() #TODO!!!!!!!!!!!!
    # Record the start time
    start_time = time.time()

    if args.input_folder:
        upscale_folder(args.input_folder, args.output_folder, args.upscale_images, args.upscale_archives, args.overwrite_existing_files, args.auto_adjust_levels, args.image_format, args.lossy_compression_quality, bool(args.use_lossless_compression))
    elif args.input_file:
        upscale_file(args.input_file, args.output_file, args.overwrite_existing_files, args.auto_adjust_levels, args.image_format, args.lossy_compression_quality, bool(args.use_lossless_compression))


    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

import sys
import functools
import os
from pathlib import Path
from ctypes import windll
import io
import cv2
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import argparse
import zipfile
import rarfile
import time
from multiprocessing import Queue, Process, Manager
os.environ["MAGICK_HOME"] = r"C:\Users\jsoos\Documents\programming\MangaJaNaiConverterGui\MangaJaNaiConverterGui\chaiNNer\ImageMagick"
from wand.image import Image as WandImage
from wand.display import display

from nodes.utils.utils import get_h_w_c
from nodes.impl.image_utils import cv_save_image, to_uint8, to_uint16
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


def mitchell_resize_wand(image, new_size):
    #print(f"mitchell_resize_wand {new_size}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
    with WandImage.from_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) as img:
        img.transform_colorspace("rgb")
        img.resize(*new_size)
        img.transform_colorspace("srgb")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


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

    # gray_image =
    return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)


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


def cv_image_is_grayscale(image):

    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the mean pixel value for each channel
    image_mean = np.mean(image, axis=(0, 1))
    gray_image_mean = np.mean(gray_image)

    # Define a threshold for considering it grayscale
    threshold = 5  # Adjust the threshold as needed

    return np.all(np.abs(image_mean - gray_image_mean) < threshold)



def _read_pil(im) -> np.ndarray | None:
    img = np.array(im)
    _, _, c = get_h_w_c(img)
    print(f"_read_pil c={c}")
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def ai_upscale_image(image, is_grayscale):
    # print(f"ai_upscale_image")
    if is_grayscale:
        if grayscale_model is not None:
            # TODO estimate vs no_tiling benchmark
            return upscale_image_node(image, grayscale_model, ESTIMATE, False)
    else:
        if color_model is not None:
            return upscale_image_node(image, color_model, ESTIMATE, False)
    return image


def postprocess_image(image):
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)


def save_image_zip(image, file_name, output_zip, image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):

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

    # resize after upscale
    if resize_height_after_upscale != 0:
        h, w = image.shape[:2]
        image = mitchell_resize_wand(image, (round(w * resize_height_after_upscale / h), resize_height_after_upscale))
    elif resize_factor_after_upscale != 100:
        h, w = image.shape[:2]
        image = mitchell_resize_wand(image, (round(w * resize_factor_after_upscale / 100), round(h * resize_factor_after_upscale / 100)))

    # Convert the resized image back to bytes
    _, buf_img = cv2.imencode(f".{image_format}", image, params)
    output_buffer = io.BytesIO(buf_img)
    upscaled_image_data = output_buffer.getvalue()

    # Add the resized image to the output zip
    output_zip.writestr(file_name, upscaled_image_data)


def save_image(image, output_file_path, image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):
    print(f"save image: {output_file_path}", flush=True)

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

    # resize after upscale
    if resize_height_after_upscale != 0:
        h, w = image.shape[:2]
        image = mitchell_resize_wand(image, (round(w * resize_height_after_upscale / h), resize_height_after_upscale))
    elif resize_factor_after_upscale != 100:
        h, w = image.shape[:2]
        image = mitchell_resize_wand(image, (round(w * resize_factor_after_upscale / 100), round(h * resize_factor_after_upscale / 100)))

    cv_save_image(output_file_path, image, params)


def preprocess_worker_archive(upscale_queue, input_archive_path, auto_adjust_levels,
resize_height_before_upscale, resize_factor_before_upscale):
    """
    given a zip or rar path, read images out of the archive, apply auto levels, add the image to upscale queue
    """

    if input_archive_path.endswith(ZIP_EXTENSIONS):
        with zipfile.ZipFile(input_archive_path, 'r') as input_zip:
            preprocess_worker_archive_file(upscale_queue, input_zip, auto_adjust_levels,
                                    resize_height_before_upscale, resize_factor_before_upscale)
    elif input_archive_path.endswith(RAR_EXTENSIONS):
        with rarfile.RarFile(input_archive_path, 'r') as input_rar:
            preprocess_worker_archive_file(upscale_queue, input_rar, auto_adjust_levels,
                                    resize_height_before_upscale, resize_factor_before_upscale)


def preprocess_worker_archive_file(upscale_queue, input_archive, auto_adjust_levels,
resize_height_before_upscale, resize_factor_before_upscale):
    """
    given an input zip or rar archive, read images out of the archive, apply auto levels, add the image to upscale queue
    """
    # print(f"preprocess_worker_archive entering aal={auto_adjust_levels}")
    # with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
        # Create a new zip file in write mode for the resized images
        #with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        # Iterate through the files in the input zip
    namelist = input_archive.namelist()
    print(f"TOTALZIP={len(namelist)}", flush=True)
    for filename in namelist:
        decoded_filename = filename
        try:
            decoded_filename = decoded_filename.encode('cp437').decode(f'cp{system_codepage}')
        except:
            pass

        # Open the file inside the input zip
        try:
            with input_archive.open(filename) as file_in_archive:
                # Read the image data
                # load_queue.put((file_in_archive.read(), filename))

                image_data = file_in_archive.read()


                # with Image.open(io.BytesIO(image_data)) as img:
                image_bytes = io.BytesIO(image_data)
                # image = _read_pil(img)
                image = _read_cv(image_bytes)

                # resize before upscale
                if resize_height_before_upscale != 0:
                    h, w = image.shape[:2]
                    image = mitchell_resize_wand(image, (round(w * resize_height_before_upscale / h), resize_height_before_upscale))
                elif resize_factor_before_upscale != 100:
                    h, w = image.shape[:2]
                    image = mitchell_resize_wand(image, (round(w * resize_factor_before_upscale / 100), round(h * resize_factor_before_upscale / 100)))

                is_grayscale = cv_image_is_grayscale(image)
                # print(f"is_grayscale? {is_grayscale} {filename}", flush=True)

                if is_grayscale and auto_adjust_levels:
                    image = enhance_contrast(image)
                else:
                    # TODO ???
                    # image = image.astype('float32')
                    # print("?!?!?!?!?")
                    # image = np.array(Image.fromarray(image).convert("L")).astype('float32')
                    pass
                upscale_queue.put((image, decoded_filename, True, is_grayscale))
        except:
            print(f"could not read as image, copying file to zip instead of upscaling: {decoded_filename}", flush=True)
            upscale_queue.put((image_data, decoded_filename, False, False))
            pass
    upscale_queue.put(SENTINEL)

    # print("preprocess_worker_archive exiting")


def preprocess_worker_folder(upscale_queue, input_folder_path, output_folder_path, output_filename, upscale_images, upscale_archives,
overwrite_existing_files, auto_adjust_levels, resize_height_before_upscale, resize_factor_before_upscale, image_format,
lossy_compression_quality, use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale):
    """
    given a folder path, recursively iterate the folder
    """
    print(f"preprocess_worker_folder entering {input_folder_path} {output_folder_path} {output_filename}", flush=True)
    for root, dirs, files in os.walk(input_folder_path):

        for filename in files:
            # for output file, create dirs if necessary, or skip if file exists and overwrite not enabled
            input_file_base = Path(filename).stem
            filename_rel = os.path.relpath(os.path.join(root, filename), input_folder_path)
            output_filename_rel = os.path.join(os.path.dirname(filename_rel), output_filename.replace('%filename%', input_file_base))
            output_file_path = Path(os.path.join(output_folder_path, output_filename_rel))


            if filename.lower().endswith(IMAGE_EXTENSIONS): # TODO if image
                if upscale_images:
                    # output_file_path = str(output_file_path.with_suffix(f'.{image_format}')).replace('%filename%', input_file_base)
                    output_file_path =  str(Path(f"{output_file_path}.{image_format}")).replace('%filename%', input_file_base) 
                    # print(f"preprocess_worker_folder overwrite={overwrite_existing_files} outfilepath={output_file_path} isfile={os.path.isfile(output_file_path)}", flush=True)
                    if not overwrite_existing_files and os.path.isfile(output_file_path):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue

                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    image = _read_cv_from_path(os.path.join(root, filename))

                    # resize before upscale
                    if resize_height_before_upscale != 0:
                        h, w = image.shape[:2]
                        image = mitchell_resize_wand(image, (round(w * resize_height_before_upscale / h), resize_height_before_upscale))
                    elif resize_factor_before_upscale != 100:
                        h, w = image.shape[:2]
                        image = mitchell_resize_wand(image, (round(w * resize_factor_before_upscale / 100), round(h * resize_factor_before_upscale / 100)))

                    is_grayscale = cv_image_is_grayscale(image)

                    if is_grayscale and auto_adjust_levels:
                        image = enhance_contrast(image)
                    else:
                        # image = np.array(Image.fromarray(image).convert("L")).astype('float32')  # TODO ???
                        pass
                    upscale_queue.put((image, output_filename_rel, True, is_grayscale))
            elif filename.lower().endswith(ZIP_EXTENSIONS):  # TODO if archive
                if upscale_archives:
                    output_file_path = str(output_file_path.with_suffix('.cbz'))
                    if not overwrite_existing_files and os.path.isfile(output_file_path):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    upscale_archive_file(os.path.join(root, filename), output_file_path,
                        auto_adjust_levels, resize_height_before_upscale, resize_factor_before_upscale, image_format,
                        lossy_compression_quality, use_lossless_compression, resize_height_after_upscale,
                        resize_factor_after_upscale) # TODO custom output extension
    upscale_queue.put(SENTINEL)
    # print("preprocess_worker_folder exiting")


def preprocess_worker_image(upscale_queue, input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels,
resize_height_before_upscale, resize_factor_before_upscale):
    """
    given an image path, apply auto levels and add to upscale queue
    """

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}", flush=True)
            return

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # with Image.open(input_image_path) as img:
        image = _read_cv_from_path(input_image_path)

        # resize before upscale
        if resize_height_before_upscale != 0:
            h, w = image.shape[:2]
            image = mitchell_resize_wand(image, (round(w * resize_height_before_upscale / h), resize_height_before_upscale))
        elif resize_factor_before_upscale != 100:
            h, w = image.shape[:2]
            image = mitchell_resize_wand(image, (round(w * resize_factor_before_upscale / 100), round(h * resize_factor_before_upscale / 100)))

        is_grayscale = cv_image_is_grayscale(image)

        if is_grayscale and auto_adjust_levels:
            image = enhance_contrast(image)

        upscale_queue.put((image, None, True, is_grayscale))
    upscale_queue.put(SENTINEL)


def upscale_worker(upscale_queue, postprocess_queue):
    """
    wait for upscale queue, for each queue entry, upscale image and add result to postprocess queue
    """
    # print("upscale_worker entering")
    while True:
        image, file_name, is_image, is_grayscale = upscale_queue.get()
        if image is None:
            break

        if is_image:
            image = ai_upscale_image(image, is_grayscale)
        postprocess_queue.put((image, file_name, is_image, is_grayscale))
    postprocess_queue.put(SENTINEL)
    # print("upscale_worker exiting")

def postprocess_worker_zip(postprocess_queue, output_zip_path, image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):
    """
    wait for postprocess queue, for each queue entry, save the image to the zip file
    """
    # print("postprocess_worker_zip entering")
    with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        while True:
            image, file_name, is_image, is_grayscale = postprocess_queue.get()
            if image is None:
                break
            if is_image:
                # image = postprocess_image(image)
                save_image_zip(image, str(Path(file_name).with_suffix(f'.{image_format}')), output_zip, image_format,
                lossy_compression_quality, use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale)
            else:  # copy file
                output_zip.writestr(file_name, image)
            print(f"PROGRESS=postprocess_worker_zip_image", flush=True)
        print(f"PROGRESS=postprocess_worker_zip_archive", flush=True)


def postprocess_worker_folder(postprocess_queue, output_folder, image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):
    """
    wait for postprocess queue, for each queue entry, save the image to the output folder
    """
    # print("postprocess_worker_folder entering")
    while True:
        image, file_name, _, _ = postprocess_queue.get()
        if image is None:
            break
        image = postprocess_image(image)
        save_image(image, os.path.join(output_folder, str(Path(f"{file_name}.{image_format}"))), image_format,
            lossy_compression_quality, use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale)
        print(f"PROGRESS=postprocess_worker_folder", flush=True)

    # print("postprocess_worker_folder exiting")


def postprocess_worker_image(postprocess_queue, output_file_path, image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):
    """
    wait for postprocess queue, for each queue entry, save the image to the output file path
    """
    while True:
        image, _, _, _ = postprocess_queue.get()
        if image is None:
            break
        # image = postprocess_image(image)

        save_image(image, output_file_path, image_format, lossy_compression_quality, use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale)
        print(f"PROGRESS=postprocess_worker_image", flush=True)


def upscale_archive_file(input_zip_path, output_zip_path, auto_adjust_levels, resize_height_before_upscale,
resize_factor_before_upscale, image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):
    # TODO accept multiple paths to reuse simple queues?

    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Process(target=preprocess_worker_archive, args=(upscale_queue, input_zip_path, auto_adjust_levels,
        resize_height_before_upscale, resize_factor_before_upscale))
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
    upscale_process.start()

    # start postprocess zip process
    postprocess_process = Process(target=postprocess_worker_zip, args=(postprocess_queue, output_zip_path, image_format, lossy_compression_quality,
        use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale))
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_image_file(input_image_path, output_image_path, overwrite_existing_files, auto_adjust_levels,
resize_height_before_upscale, resize_factor_before_upscale, image_format, lossy_compression_quality,
use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale):

    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess image process
    preprocess_process = Process(target=preprocess_worker_image, args=(upscale_queue, input_image_path,
        output_image_path, overwrite_existing_files, auto_adjust_levels, resize_height_before_upscale,
        resize_factor_before_upscale))
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
    upscale_process.start()

    # start postprocess image process
    postprocess_process = Process(target=postprocess_worker_image, args=(postprocess_queue, output_image_path,
        image_format, lossy_compression_quality, use_lossless_compression, resize_height_after_upscale,
        resize_factor_after_upscale))
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_file(input_file_path, output_folder_path, output_filename, overwrite_existing_files,
auto_adjust_levels, resize_height_before_upscale, resize_factor_before_upscale,
image_format, lossy_compression_quality, use_lossless_compression,
resize_height_after_upscale, resize_factor_after_upscale):

    input_file_base = Path(input_file_path).stem


    if input_file_path.lower().endswith(ARCHIVE_EXTENSIONS):

        output_file_path = str(Path(os.path.join(output_folder_path, output_filename.replace('%filename%', input_file_base))).with_suffix('.cbz'))
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_archive_file(input_file_path, output_file_path, auto_adjust_levels,
            resize_height_before_upscale, resize_factor_before_upscale,
            image_format, lossy_compression_quality, use_lossless_compression,
            resize_height_after_upscale, resize_factor_after_upscale)

    elif input_file_path.lower().endswith(IMAGE_EXTENSIONS):

        output_file_path = str(Path(os.path.join(output_folder_path, output_filename.replace('%filename%', input_file_base))).with_suffix(f'.{image_format}'))
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_image_file(input_file_path, output_file_path, overwrite_existing_files,
            auto_adjust_levels, resize_height_before_upscale, resize_factor_before_upscale,
            image_format, lossy_compression_quality, use_lossless_compression,
            resize_height_after_upscale, resize_factor_after_upscale)


def upscale_folder(input_folder_path, output_folder_path, output_filename, upscale_images, upscale_archives, overwrite_existing_files,
auto_adjust_levels, resize_height_before_upscale, resize_factor_before_upscale, image_format,
lossy_compression_quality, use_lossless_compression, resize_height_after_upscale, resize_factor_after_upscale):
    # print("upscale_folder: entering")

    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess folder process
    preprocess_process = Process(target=preprocess_worker_folder, args=(upscale_queue, input_folder_path, output_folder_path,
        output_filename, upscale_images, upscale_archives, overwrite_existing_files, auto_adjust_levels, resize_height_before_upscale,
        resize_factor_before_upscale, image_format, lossy_compression_quality, use_lossless_compression,
        resize_height_after_upscale, resize_factor_after_upscale))
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(target=upscale_worker, args=(upscale_queue, postprocess_queue))
    upscale_process.start()

    # start postprocess folder process
    postprocess_process = Process(target=postprocess_worker_folder, args=(postprocess_queue, output_folder_path,
        image_format, lossy_compression_quality, use_lossless_compression, resize_height_after_upscale,
        resize_factor_after_upscale))
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()

sys.stdout.reconfigure(encoding='utf-8')
parser = argparse.ArgumentParser()

parser.add_argument('--input-file-path', required=False)
parser.add_argument('--output-filename', required=False)
parser.add_argument('--input-folder-path', required=False)
parser.add_argument('--output-folder-path', required=False)
parser.add_argument('--upscale-archives', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--upscale-images', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--overwrite-existing-files', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--auto-adjust-levels', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--resize-height-before-upscale', required=False, type=int, default=0)
parser.add_argument('--resize-factor-before-upscale', required=False, type=float, default=100)
parser.add_argument('--image-format')
parser.add_argument('--lossy-compression-quality')
parser.add_argument('--use-lossless-compression', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--grayscale-model-path', required=False)
parser.add_argument('--color-model-path', required=False)
parser.add_argument('--resize-height-after-upscale', required=False, type=int, default=0)
parser.add_argument('--resize-factor-after-upscale', required=False, type=float, default=100)

args = parser.parse_args()
print(args)


SENTINEL = (None, None, None, None)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
ZIP_EXTENSIONS = ('.zip', '.cbz')
RAR_EXTENSIONS = ('.rar', '.cbr')
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + RAR_EXTENSIONS
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

    if args.input_folder_path:
        upscale_folder(args.input_folder_path, args.output_folder_path, args.output_filename, args.upscale_images, args.upscale_archives,
            args.overwrite_existing_files, args.auto_adjust_levels, args.resize_height_before_upscale,
            args.resize_factor_before_upscale, args.image_format, args.lossy_compression_quality,
            args.use_lossless_compression, args.resize_height_after_upscale, args.resize_factor_after_upscale)
    elif args.input_file_path:
        upscale_file(args.input_file_path, args.output_folder_path, args.output_filename, args.overwrite_existing_files, args.auto_adjust_levels,
            args.resize_height_before_upscale, args.resize_factor_before_upscale, args.image_format,
            args.lossy_compression_quality, args.use_lossless_compression, args.resize_height_after_upscale,
            args.resize_factor_after_upscale)


    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

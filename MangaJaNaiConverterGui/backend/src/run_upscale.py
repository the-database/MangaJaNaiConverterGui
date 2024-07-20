import sys
import json
import os
import platform
from pathlib import Path
import ctypes
import io
import cv2
import pillow_avif  # noqa: F401
from PIL import Image, ImageFilter, ImageCms
import numpy as np
import argparse
import zipfile
import rarfile
import time
from typing import Callable
from multiprocess import Queue, Process, set_start_method
from chainner_ext import resize, ResizeFilter

sys.path.append(
    os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
)

from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
from api import (
    NodeContext,
    SettingsParser,
)
from progress_controller import ProgressController, ProgressToken
from nodes.utils.utils import get_h_w_c
from nodes.impl.image_utils import cv_save_image, to_uint8, to_uint16, normalize
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
    upscale_image_node,
)
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    NO_TILING,
    MAX_TILE_SIZE,
    TileSize,
)


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self, progress: ProgressToken, settings: SettingsParser, storage_dir: Path
    ) -> None:
        super().__init__()

        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir

        self.cleanup_fns: set[Callable[[], None]] = set()

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0.001)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

        # TODO: send progress event

    @property
    def settings(self) -> SettingsParser:
        """
        Returns the settings of the current node execution.
        """
        return self.__settings

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def add_cleanup(self, fn: Callable[[], None]) -> None:
        self.cleanup_fns.add(fn)


def get_tile_size(tile_size_str):
    if tile_size_str == "Auto (Estimate)":
        return ESTIMATE
    elif tile_size_str == "Maximum":
        return MAX_TILE_SIZE
    elif tile_size_str == "No Tiling":
        return NO_TILING
    elif tile_size_str.isdecimal():
        return TileSize(int(tile_size_str))

    return ESTIMATE


"""
lanczos downscale without color conversion, for pre-upscale
downscale and final color downscale 
"""


def standard_resize(image, new_size):
    new_image = np.float32(image) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.Lanczos, False)
    return (new_image * 255).round().astype(np.uint8)


"""
final downscale for grayscale images only
"""


def dotgain20_resize(image, new_size):
    h, w = image.shape[:2]
    size_ratio = h / new_size[1]
    blur_size = (1 / size_ratio - 1) / 3.5
    if blur_size >= 0.1:
        if blur_size > 250:
            blur_size = 250

    pil_image = Image.fromarray(image[:, :, 0], mode="L")
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))
    pil_image = ImageCms.applyTransform(pil_image, dotgain20togamma1transform, False)

    new_image = np.array(pil_image)
    new_image = np.float32(new_image) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.CubicCatrom, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    pil_image = Image.fromarray(new_image[:, :, 0], mode="L")
    pil_image = ImageCms.applyTransform(pil_image, gamma1todotgain20transform, False)
    return np.array(pil_image)


def image_resize(image, new_size, is_grayscale):
    if is_grayscale:
        return dotgain20_resize(image, new_size)

    return standard_resize(image, new_size)


def get_system_codepage():
    return None if is_linux else ctypes.windll.kernel32.GetConsoleOutputCP()


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

    print(
        f"Auto adjusted levels: new black level = {new_black_level}; new white level = {new_white_level}",
        flush=True,
    )

    image_array = np.array(image_p).astype("float32")
    image_array = np.maximum(image_array - new_black_level, 0) / (
        new_white_level - new_black_level
    )
    image_array = np.clip(image_array, 0, 1)

    # gray_image =
    return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)


def _read_cv(img_stream):
    return cv2.imdecode(
        np.frombuffer(img_stream.read(), dtype=np.uint8), cv2.IMREAD_COLOR
    )


def _read_cv_from_path(path):
    img = None
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as cv_err:
        print(f"Error loading image, trying with imdecode: {cv_err}", flush=True)

    if img is None:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(
                f'Error reading image image from path "{path}". Image may be corrupt.'
            ) from e

    if img is None:  # type: ignore
        raise RuntimeError(
            f'Error reading image image from path "{path}". Image may be corrupt.'
        )

    return img


def _read_image(img_stream, filename):
    for extension in CV2_IMAGE_EXTENSIONS:
        if filename.lower().endswith(extension):
            return _read_cv(img_stream)
    return _read_pil(img_stream)


def _read_image_from_path(path):
    for extension in CV2_IMAGE_EXTENSIONS:
        if path.lower().endswith(extension):
            return _read_cv_from_path(path)

    return _read_pil_from_path(path)


def _read_pil(img_stream):
    im = Image.open(img_stream, formats=["AVIF"])
    return _pil_to_cv2(im)


def _read_pil_from_path(path):
    im = Image.open(path)
    return _pil_to_cv2(im)


def _pil_to_cv2(im):
    img = np.array(im)
    _, _, c = get_h_w_c(img)
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def cv_image_is_grayscale(image):
    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    b, g, r = cv2.split(image[:, :, :3])

    ignore_threshold = 12

    # getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)

    # create masks to identify pure black and pure white pixels
    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))

    # combine masks to exclude both pure black and pure white pixels
    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    # exclude pure black and pure white pixels from diff_sum and image size calculation
    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    # if the entire image is pure black or pure white, return False
    if size_without_black_and_white == 0:
        return False

    # finding ratio of diff_sum with respect to size of image without pure black and pure white pixels
    ratio = diff_sum / size_without_black_and_white

    return ratio < 1


def get_chain_for_image(image, target_scale, target_width, target_height, chains):
    original_height, original_width, _ = get_h_w_c(image)

    if target_width != 0 and target_height != 0:
        target_scale = min(
            target_height / original_height, target_width / original_width
        )
    if target_height != 0:
        target_scale = target_height / original_height
    elif target_width != 0:
        target_scale = target_width / original_width

    is_grayscale = cv_image_is_grayscale(image)

    for chain in chains:
        if should_chain_activate_for_image(
            original_width, original_height, is_grayscale, target_scale, chain
        ):
            print("Matched Chain:", chain)
            return chain, is_grayscale, original_width, original_height

    return None, None, None, None


def should_chain_activate_for_image(
    original_width, original_height, is_grayscale, target_scale, chain
):
    min_width, min_height = [int(x) for x in chain["MinResolution"].split("x")]
    max_width, max_height = [int(x) for x in chain["MaxResolution"].split("x")]

    # resolution tests
    if min_width != 0 and min_width > original_width:
        return False
    if min_height != 0 and min_height > original_height:
        return False
    if max_width != 0 and max_width < original_width:
        return False
    if max_height != 0 and max_height < original_height:
        return False

    # color / grayscale tests
    if is_grayscale and not chain["IsGrayscale"]:
        return False
    if not is_grayscale and not chain["IsColor"]:
        return False

    # scale tests
    if chain["MaxScaleFactor"] != 0 and target_scale > chain["MaxScaleFactor"]:
        return False
    if chain["MinScaleFactor"] != 0 and target_scale < chain["MinScaleFactor"]:
        return False

    return True


def ai_upscale_image(image, model_tile_size, model):
    global context
    if model is not None:
        return upscale_image_node(
            context, image, model, False, 0, model_tile_size, None, False
        )
    return image


def postprocess_image(image):
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)


def final_target_resize(
    image,
    target_scale,
    target_width,
    target_height,
    original_width,
    original_height,
    is_grayscale,
):
    # fit to dimensions
    if target_height != 0 and target_width != 0:
        h, w = image.shape[:2]
        # determine whether to fit to height or width
        if target_height / original_height < target_width / original_width:
            target_width = 0
        else:
            target_height = 0

    # resize height, keep proportional width
    if target_height != 0:
        h, w = image.shape[:2]
        if h != target_height:
            return image_resize(
                image, (round(w * target_height / h), target_height), is_grayscale
            )
    # resize width, keep proportional height
    elif target_width != 0:
        h, w = image.shape[:2]
        if w != target_width:
            return image_resize(
                image, (target_width, round(h * target_width / w)), is_grayscale
            )
    else:
        h, w = image.shape[:2]
        new_target_height = original_height * target_scale
        if h != new_target_height:
            return image_resize(
                image,
                (round(w * new_target_height / h), new_target_height),
                is_grayscale,
            )

    return image


def save_image_zip(
    image,
    file_name,
    output_zip,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    original_width,
    original_height,
    target_scale,
    target_width,
    target_height,
    is_grayscale,
):
    print(f"save image to zip: {file_name}", flush=True)

    if image_format == "avif":
        image = to_uint8(image, normalized=True)
        channels = get_h_w_c(image)[2]
        if channels == 1:
            pass
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        image = final_target_resize(
            image,
            target_scale,
            target_width,
            target_height,
            original_width,
            original_height,
            is_grayscale,
        )

        with Image.fromarray(image) as pil_im:
            output_buffer = io.BytesIO()
            pil_im.save(output_buffer, format=image_format)
    else:
        if image_format == "jpg":
            params = [
                cv2.IMWRITE_JPEG_QUALITY,
                int(lossy_compression_quality),
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                1,  # jpeg_progressive
            ]
        elif image_format == "webp":
            params = [
                cv2.IMWRITE_WEBP_QUALITY,
                101 if use_lossless_compression else int(lossy_compression_quality),
            ]
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

        image = final_target_resize(
            image,
            target_scale,
            target_width,
            target_height,
            original_width,
            original_height,
            is_grayscale,
        )

        # Convert the resized image back to bytes
        _, buf_img = cv2.imencode(f".{image_format}", image, params)
        output_buffer = io.BytesIO(buf_img)

    upscaled_image_data = output_buffer.getvalue()

    # Add the resized image to the output zip
    output_zip.writestr(file_name, upscaled_image_data)


def save_image(
    image,
    output_file_path,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    original_width,
    original_height,
    target_scale,
    target_width,
    target_height,
    is_grayscale,
):
    print(f"save image: {output_file_path}", flush=True)

    # save with pillow
    if image_format == "avif":
        image = to_uint8(image, normalized=True)
        channels = get_h_w_c(image)[2]
        if channels == 1:
            pass
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        image = final_target_resize(
            image,
            target_scale,
            target_width,
            target_height,
            original_width,
            original_height,
            is_grayscale,
        )

        with Image.fromarray(image) as pil_im:
            pil_im.save(output_file_path, quality=lossy_compression_quality)
    else:
        # save with cv2
        if image_format == "jpg":
            params = [
                cv2.IMWRITE_JPEG_QUALITY,
                int(lossy_compression_quality),
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                1,  # jpeg_progressive
            ]
        elif image_format == "webp":
            params = [
                cv2.IMWRITE_WEBP_QUALITY,
                101 if use_lossless_compression else int(lossy_compression_quality),
            ]
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

        image = final_target_resize(
            image,
            target_scale,
            target_width,
            target_height,
            original_width,
            original_height,
            is_grayscale,
        )

        cv_save_image(output_file_path, image, params)


def preprocess_worker_archive(
    upscale_queue,
    input_archive_path,
    output_archive_path,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    """
    given a zip or rar path, read images out of the archive, apply auto levels, add the image to upscale queue
    """

    if input_archive_path.endswith(ZIP_EXTENSIONS):
        with zipfile.ZipFile(input_archive_path, "r") as input_zip:
            preprocess_worker_archive_file(
                upscale_queue,
                input_zip,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
            )
    elif input_archive_path.endswith(RAR_EXTENSIONS):
        with rarfile.RarFile(input_archive_path, "r") as input_rar:
            preprocess_worker_archive_file(
                upscale_queue,
                input_rar,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
            )


def preprocess_worker_archive_file(
    upscale_queue,
    input_archive,
    output_archive_path,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    """
    given an input zip or rar archive, read images out of the archive, apply auto levels, add the image to upscale queue
    """
    os.makedirs(os.path.dirname(output_archive_path), exist_ok=True)
    namelist = input_archive.namelist()
    print(f"TOTALZIP={len(namelist)}", flush=True)
    for filename in namelist:
        decoded_filename = filename
        try:
            decoded_filename = decoded_filename.encode("cp437").decode(
                f"cp{system_codepage}"
            )
        except:  # noqa: E722
            pass

        # Open the file inside the input zip
        try:
            with input_archive.open(filename) as file_in_archive:
                # Read the image data

                image_data = file_in_archive.read()

                image_bytes = io.BytesIO(image_data)
                image = _read_image(image_bytes, filename)

                chain, is_grayscale, original_width, original_height = (
                    get_chain_for_image(
                        image, target_scale, target_width, target_height, chains
                    )
                )
                model = None
                if chain is not None:
                    resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                    resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
                    resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

                    # resize width and height, distorting image
                    if (
                        resize_height_before_upscale != 0
                        and resize_width_before_upscale != 0
                    ):
                        h, w = image.shape[:2]
                        image = standard_resize(
                            image,
                            (resize_width_before_upscale, resize_height_before_upscale),
                        )
                    # resize height, keep proportional width
                    elif resize_height_before_upscale != 0:
                        h, w = image.shape[:2]
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_height_before_upscale / h),
                                resize_height_before_upscale,
                            ),
                        )
                    # resize width, keep proportional height
                    elif resize_width_before_upscale != 0:
                        h, w = image.shape[:2]
                        image = standard_resize(
                            image,
                            (
                                resize_width_before_upscale,
                                round(h * resize_width_before_upscale / w),
                            ),
                        )
                    elif resize_factor_before_upscale != 100:
                        h, w = image.shape[:2]
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_factor_before_upscale / 100),
                                round(h * resize_factor_before_upscale / 100),
                            ),
                        )

                    if is_grayscale and chain["AutoAdjustLevels"]:
                        image = enhance_contrast(image)
                    else:
                        image = normalize(image)

                    model_abs_path = get_model_abs_path(chain['ModelFilePath'])

                    if model_abs_path in loaded_models:
                        model = loaded_models[model_abs_path]

                    elif os.path.exists(model_abs_path):
                        model, dirname, basename = load_model_node(
                            context, Path(model_abs_path)
                        )
                        if model is not None:
                            loaded_models[model_abs_path] = model
                else:
                    image = normalize(image)

                image = np.ascontiguousarray(image)

                upscale_queue.put(
                    (
                        image,
                        decoded_filename,
                        True,
                        is_grayscale,
                        original_width,
                        original_height,
                        get_tile_size(chain["ModelTileSize"]),
                        model,
                    )
                )
        except Exception as e:
            print(
                f"could not read as image, copying file to zip instead of upscaling: {decoded_filename}, {e}",
                flush=True,
            )
            upscale_queue.put(
                (image_data, decoded_filename, False, False, None, None, None, None)
            )
        #     pass
    upscale_queue.put(UPSCALE_SENTINEL)

    # print("preprocess_worker_archive exiting")


def preprocess_worker_folder(
    upscale_queue,
    input_folder_path,
    output_folder_path,
    output_filename,
    upscale_images,
    upscale_archives,
    overwrite_existing_files,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    """
    given a folder path, recursively iterate the folder
    """
    print(
        f"preprocess_worker_folder entering {input_folder_path} {output_folder_path} {output_filename}",
        flush=True,
    )
    for root, dirs, files in os.walk(input_folder_path):
        for filename in files:
            # for output file, create dirs if necessary, or skip if file exists and overwrite not enabled
            input_file_base = Path(filename).stem
            filename_rel = os.path.relpath(
                os.path.join(root, filename), input_folder_path
            )
            output_filename_rel = os.path.join(
                os.path.dirname(filename_rel),
                output_filename.replace("%filename%", input_file_base),
            )
            output_file_path = Path(
                os.path.join(output_folder_path, output_filename_rel)
            )

            if filename.lower().endswith(IMAGE_EXTENSIONS):  # TODO if image
                if upscale_images:
                    output_file_path = str(
                        Path(f"{output_file_path}.{image_format}")
                    ).replace("%filename%", input_file_base)

                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue

                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    image = _read_image_from_path(os.path.join(root, filename))

                    chain, is_grayscale, original_width, original_height = (
                        get_chain_for_image(
                            image, target_scale, target_width, target_height, chains
                        )
                    )
                    model = None
                    if chain is not None:
                        resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                        resize_height_before_upscale = chain[
                            "ResizeHeightBeforeUpscale"
                        ]
                        resize_factor_before_upscale = chain[
                            "ResizeFactorBeforeUpscale"
                        ]

                        # resize width and height, distorting image
                        if (
                            resize_height_before_upscale != 0
                            and resize_width_before_upscale != 0
                        ):
                            h, w = image.shape[:2]
                            image = standard_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    resize_height_before_upscale,
                                ),
                            )
                        # resize height, keep proportional width
                        elif resize_height_before_upscale != 0:
                            h, w = image.shape[:2]
                            image = standard_resize(
                                image,
                                (
                                    round(w * resize_height_before_upscale / h),
                                    resize_height_before_upscale,
                                ),
                            )
                        # resize width, keep proportional height
                        elif resize_width_before_upscale != 0:
                            h, w = image.shape[:2]
                            image = standard_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    round(h * resize_width_before_upscale / w),
                                ),
                            )
                        elif resize_factor_before_upscale != 100:
                            h, w = image.shape[:2]
                            image = standard_resize(
                                image,
                                (
                                    round(w * resize_factor_before_upscale / 100),
                                    round(h * resize_factor_before_upscale / 100),
                                ),
                            )

                        if is_grayscale and chain["AutoAdjustLevels"]:
                            image = enhance_contrast(image)
                        else:
                            image = normalize(image)

                        model_abs_path = get_model_abs_path(chain['ModelFilePath'])

                        if model_abs_path in loaded_models:
                            model = loaded_models[model_abs_path]

                        elif os.path.exists(model_abs_path):
                            model, dirname, basename = load_model_node(
                                context, Path(model_abs_path)
                            )
                            if model is not None:
                                loaded_models[model_abs_path] = model
                    else:
                        image = normalize(image)

                    image = np.ascontiguousarray(image)

                    upscale_queue.put(
                        (
                            image,
                            output_filename_rel,
                            True,
                            is_grayscale,
                            original_width,
                            original_height,
                            get_tile_size(chain["ModelTileSize"]),
                            model,
                        )
                    )
            elif filename.lower().endswith(ZIP_EXTENSIONS):  # TODO if archive
                if upscale_archives:
                    output_file_path = str(output_file_path.with_suffix(".cbz"))
                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    upscale_archive_file(
                        os.path.join(root, filename),
                        output_file_path,
                        image_format,
                        lossy_compression_quality,
                        use_lossless_compression,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        loaded_models,
                    )  # TODO custom output extension
    upscale_queue.put(UPSCALE_SENTINEL)
    # print("preprocess_worker_folder exiting")


def preprocess_worker_image(
    upscale_queue,
    input_image_path,
    output_image_path,
    overwrite_existing_files,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    """
    given an image path, apply auto levels and add to upscale queue
    """
    global context

    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}", flush=True)
            return

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # with Image.open(input_image_path) as img:
        image = _read_image_from_path(input_image_path)

        chain, is_grayscale, original_width, original_height = get_chain_for_image(
            image, target_scale, target_width, target_height, chains
        )
        model = None
        if chain is not None:
            resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
            resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
            resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

            # resize width and height, distorting image
            if resize_height_before_upscale != 0 and resize_width_before_upscale != 0:
                h, w = image.shape[:2]
                image = standard_resize(
                    image, (resize_width_before_upscale, resize_height_before_upscale)
                )
            # resize height, keep proportional width
            elif resize_height_before_upscale != 0:
                h, w = image.shape[:2]
                image = standard_resize(
                    image,
                    (
                        round(w * resize_height_before_upscale / h),
                        resize_height_before_upscale,
                    ),
                )
            # resize width, keep proportional height
            elif resize_width_before_upscale != 0:
                h, w = image.shape[:2]
                image = standard_resize(
                    image,
                    (
                        resize_width_before_upscale,
                        round(h * resize_width_before_upscale / w),
                    ),
                )
            elif resize_factor_before_upscale != 100:
                h, w = image.shape[:2]
                image = standard_resize(
                    image,
                    (
                        round(w * resize_factor_before_upscale / 100),
                        round(h * resize_factor_before_upscale / 100),
                    ),
                )

            if is_grayscale and chain["AutoAdjustLevels"]:
                image = enhance_contrast(image)
            else:
                image = normalize(image)

            model_abs_path = get_model_abs_path(chain['ModelFilePath'])

            if model_abs_path in loaded_models:
                model = loaded_models[model_abs_path]

            elif os.path.exists(model_abs_path):
                model, dirname, basename = load_model_node(
                    context, Path(model_abs_path)
                )
                if model is not None:
                    loaded_models[model_abs_path] = model
        else:
            print("No chain!!!!!!!")
            image = normalize(image)

        image = np.ascontiguousarray(image)

        upscale_queue.put(
            (
                image,
                None,
                True,
                is_grayscale,
                original_width,
                original_height,
                get_tile_size(chain["ModelTileSize"]),
                model,
            )
        )
    upscale_queue.put(UPSCALE_SENTINEL)


def upscale_worker(upscale_queue, postprocess_queue):
    """
    wait for upscale queue, for each queue entry, upscale image and add result to postprocess queue
    """
    # print("upscale_worker entering")
    while True:
        (
            image,
            file_name,
            is_image,
            is_grayscale,
            original_width,
            original_height,
            model_tile_size,
            model,
        ) = upscale_queue.get()
        if image is None:
            break

        if is_image:
            image = ai_upscale_image(image, model_tile_size, model)
        postprocess_queue.put(
            (image, file_name, is_image, is_grayscale, original_width, original_height)
        )
    postprocess_queue.put(POSTPROCESS_SENTINEL)
    # print("upscale_worker exiting")


def postprocess_worker_zip(
    postprocess_queue,
    output_zip_path,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
):
    """
    wait for postprocess queue, for each queue entry, save the image to the zip file
    """
    # print("postprocess_worker_zip entering")
    with zipfile.ZipFile(output_zip_path, "w") as output_zip:
        while True:
            (
                image,
                file_name,
                is_image,
                is_grayscale,
                original_width,
                original_height,
            ) = postprocess_queue.get()
            if image is None:
                break
            if is_image:
                # image = postprocess_image(image)
                save_image_zip(
                    image,
                    str(Path(file_name).with_suffix(f".{image_format}")),
                    output_zip,
                    image_format,
                    lossy_compression_quality,
                    use_lossless_compression,
                    original_width,
                    original_height,
                    target_scale,
                    target_width,
                    target_height,
                    is_grayscale,
                )
            else:  # copy file
                output_zip.writestr(file_name, image)
            print("PROGRESS=postprocess_worker_zip_image", flush=True)
        print("PROGRESS=postprocess_worker_zip_archive", flush=True)


def postprocess_worker_folder(
    postprocess_queue,
    output_folder_path,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
):
    """
    wait for postprocess queue, for each queue entry, save the image to the output folder
    """
    # print("postprocess_worker_folder entering")
    while True:
        image, file_name, _, is_grayscale, original_width, original_height = (
            postprocess_queue.get()
        )
        if image is None:
            break
        image = postprocess_image(image)
        save_image(
            image,
            os.path.join(output_folder_path, str(Path(f"{file_name}.{image_format}"))),
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
        )
        print("PROGRESS=postprocess_worker_folder", flush=True)

    # print("postprocess_worker_folder exiting")


def postprocess_worker_image(
    postprocess_queue,
    output_file_path,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
):
    """
    wait for postprocess queue, for each queue entry, save the image to the output file path
    """
    while True:
        image, _, _, is_grayscale, original_width, original_height = (
            postprocess_queue.get()
        )
        if image is None:
            break
        # image = postprocess_image(image)

        save_image(
            image,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
        )
        print("PROGRESS=postprocess_worker_image", flush=True)


def upscale_archive_file(
    input_zip_path,
    output_zip_path,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    # TODO accept multiple paths to reuse simple queues?

    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Process(
        target=preprocess_worker_archive,
        args=(
            upscale_queue,
            input_zip_path,
            output_zip_path,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess zip process
    postprocess_process = Process(
        target=postprocess_worker_zip,
        args=(
            postprocess_queue,
            output_zip_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_image_file(
    input_image_path,
    output_image_path,
    overwrite_existing_files,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess image process
    preprocess_process = Process(
        target=preprocess_worker_image,
        args=(
            upscale_queue,
            input_image_path,
            output_image_path,
            overwrite_existing_files,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess image process
    postprocess_process = Process(
        target=postprocess_worker_image,
        args=(
            postprocess_queue,
            output_image_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_file(
    input_file_path,
    output_folder_path,
    output_filename,
    overwrite_existing_files,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    input_file_base = Path(input_file_path).stem

    if input_file_path.lower().endswith(ARCHIVE_EXTENSIONS):
        output_file_path = str(
            Path(
                os.path.join(
                    output_folder_path,
                    output_filename.replace("%filename%", input_file_base),
                )
            ).with_suffix(".cbz")
        )
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_archive_file(
            input_file_path,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
        )

    elif input_file_path.lower().endswith(IMAGE_EXTENSIONS):
        output_file_path = str(
            Path(
                os.path.join(
                    output_folder_path,
                    output_filename.replace("%filename%", input_file_base),
                )
            ).with_suffix(f".{image_format}")
        )
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_image_file(
            input_file_path,
            output_file_path,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
        )


def upscale_folder(
    input_folder_path,
    output_folder_path,
    output_filename,
    upscale_images,
    upscale_archives,
    overwrite_existing_files,
    image_format,
    lossy_compression_quality,
    use_lossless_compression,
    target_scale,
    target_width,
    target_height,
    chains,
    loaded_models,
):
    # print("upscale_folder: entering")

    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = Queue(maxsize=1)

    # start preprocess folder process
    preprocess_process = Process(
        target=preprocess_worker_folder,
        args=(
            upscale_queue,
            input_folder_path,
            output_folder_path,
            output_filename,
            upscale_images,
            upscale_archives,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Process(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess folder process
    postprocess_process = Process(
        target=postprocess_worker_folder,
        args=(
            postprocess_queue,
            output_folder_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def get_model_abs_path(chain_model_file_path):
    relative_path = os.path.join("../../models", chain_model_file_path) if is_linux \
        else os.path.join("./models", chain_model_file_path)

    return os.path.abspath(relative_path)

def get_gamma_icc_profile():
    profile_path = '../../ImageMagick/Custom Gray Gamma 1.0.icc' if is_linux else r'.\ImageMagick\Custom Gray Gamma 1.0.icc'
    return ImageCms.getOpenProfile(profile_path)

def get_dot20_icc_profile():
    profile_path = '../../ImageMagick/Dot Gain 20%.icc' if is_linux else r'.\ImageMagick\Dot Gain 20%.icc'
    return ImageCms.getOpenProfile(profile_path)


is_linux = platform.system() == "Linux"
if is_linux:
    set_start_method('spawn', force=True)


sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()

parser.add_argument("--settings", required=True)

args = parser.parse_args()

with open(args.settings, mode="r", encoding="utf-8") as f:
    settings = json.load(f)

workflow = settings["Workflows"]["$values"][settings["SelectedWorkflowIndex"]]

UPSCALE_SENTINEL = (None, None, None, None, None, None, None, None)
POSTPROCESS_SENTINEL = (None, None, None, None, None, None)
CV2_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_EXTENSIONS = CV2_IMAGE_EXTENSIONS + tuple(".avif")
ZIP_EXTENSIONS = (".zip", ".cbz")
RAR_EXTENSIONS = (".rar", ".cbr")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + RAR_EXTENSIONS
loaded_models = {}
system_codepage = get_system_codepage()

settings = SettingsParser(
    {
        "use_cpu": settings["UseCpu"],
        "use_fp16": settings["UseFp16"],
        "gpu_index": settings["SelectedDeviceIndex"],
        "budget_limit": 0,
    }
)

context = _ExecutorNodeContext(ProgressController(), settings, None)

gamma1icc = get_gamma_icc_profile()
dotgain20icc = get_dot20_icc_profile()

dotgain20togamma1transform = ImageCms.buildTransformFromOpenProfiles(
    dotgain20icc, gamma1icc, "L", "L"
)
gamma1todotgain20transform = ImageCms.buildTransformFromOpenProfiles(
    gamma1icc, dotgain20icc, "L", "L"
)

if __name__ == "__main__":
    # gc.disable() #TODO!!!!!!!!!!!!
    # Record the start time
    start_time = time.time()

    image_format = None
    if workflow["WebpSelected"]:
        image_format = "webp"
    elif workflow["PngSelected"]:
        image_format = "png"
    elif workflow["AvifSelected"]:
        image_format = "avif"
    else:
        image_format = "jpeg"

    target_scale = None
    target_width = 0
    target_height = 0

    if workflow["ModeScaleSelected"]:
        target_scale = workflow["UpscaleScaleFactor"]
    elif workflow["ModeWidthSelected"]:
        target_width = workflow["ResizeWidthAfterUpscale"]
    elif workflow["ModeHeightSelected"]:
        target_height = workflow["ResizeHeightAfterUpscale"]
    else:
        target_width = workflow["DisplayDeviceWidth"]
        target_height = workflow["DisplayDeviceHeight"]

    if workflow["SelectedTabIndex"] == 1:
        upscale_folder(
            workflow["InputFolderPath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["UpscaleImages"],
            workflow["UpscaleArchives"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
        )
    elif workflow["SelectedTabIndex"] == 0:
        upscale_file(
            workflow["InputFilePath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
        )

    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

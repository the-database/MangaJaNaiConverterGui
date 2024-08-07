from __future__ import annotations

import base64

import cv2
import navi
import numpy as np

from api import BaseOutput, BroadcastData, InputId, OutputKind

from ...impl.image_utils import normalize, to_uint8
from ...impl.resize import ResizeFilter, resize
from ...utils.format import format_image_with_channels
from ...utils.utils import get_h_w_c, round_half_up


class NumPyOutput(BaseOutput[np.ndarray]):
    """Output a NumPy array"""

    def __init__(
        self,
        output_type: navi.ExpressionJson,
        label: str,
        kind: OutputKind = "generic",
        has_handle: bool = True,
    ) -> None:
        super().__init__(
            output_type,
            label,
            kind=kind,
            has_handle=has_handle,
            associated_type=np.ndarray,
        )

    def enforce(self, value: object) -> np.ndarray:
        assert isinstance(value, np.ndarray)
        return value


def AudioOutput():
    """Output a 1D Audio NumPy array"""
    return NumPyOutput("Audio", "Audio")


class ImageOutput(NumPyOutput):
    def __init__(
        self,
        label: str = "Image",
        *,
        image_type: navi.ExpressionJson = "Image",
        kind: OutputKind = "generic",
        has_handle: bool = True,
        channels: int | None = None,
        shape_as: int | InputId | None = None,
        size_as: int | InputId | None = None,
        assume_normalized: bool = False,
    ) -> None:
        # narrow down type
        if channels is not None:
            image_type = navi.intersect_with_error(
                image_type, navi.Image(channels=channels)
            )
        if shape_as is not None:
            image_type = navi.intersect_with_error(image_type, f"Input{shape_as}")
        if size_as is not None:
            image_type = navi.intersect_with_error(
                image_type, navi.Image(size_as=f"Input{size_as}")
            )

        super().__init__(image_type, label, kind=kind, has_handle=has_handle)

        self.channels: int | None = channels
        self.assume_normalized: bool = assume_normalized

        if shape_as is not None:
            self.as_passthrough_of(shape_as)

    def get_broadcast_data(self, value: np.ndarray) -> BroadcastData:
        h, w, c = get_h_w_c(value)
        return {
            "height": h,
            "width": w,
            "channels": c,
        }

    def get_broadcast_type(self, value: np.ndarray):
        h, w, c = get_h_w_c(value)
        return navi.Image(width=w, height=h, channels=c)

    def enforce(self, value: object) -> np.ndarray:
        assert isinstance(value, np.ndarray)

        h, w, c = get_h_w_c(value)

        if h == 0 or w == 0:
            raise ValueError(
                f"The output {self.label} returned an empty image (w={w} h={h})."
                f" This is a bug in the implementation of the node."
                f" Please report this bug."
            )

        if self.channels is not None and c != self.channels:
            expected = format_image_with_channels([self.channels])
            actual = format_image_with_channels([c])
            raise ValueError(
                f"The output {self.label} was supposed to return {expected} but actually returned {actual}."
                f" This is a bug in the implementation of the node."
                f" Please report this bug."
            )

        # flatting 3D single-channel images to 2D
        if c == 1 and value.ndim == 3:
            value = value[:, :, 0]

        if not self.assume_normalized:
            value = normalize(value)

        assert value.dtype == np.float32, (
            f"The output {self.label} did not return a normalized image."
            f" This is a bug in the implementation of the node."
            f" Please report this bug."
            f"\n\nTo the author of this node: Either use `normalize` or remove `assume_normalized=True` from this output."
        )

        # make image readonly
        value.setflags(write=False)

        return value


def preview_encode(
    img: np.ndarray,
    target_size: int = 512,
    grace: float = 1.2,
    lossless: bool = False,
) -> tuple[str, np.ndarray]:
    """
    resize the image, so the preview loads faster and doesn't lag the UI
    512 was chosen as the default target because a 512x512 RGBA 8bit PNG is at most 1MB in size
    """
    h, w, c = get_h_w_c(img)

    max_size = target_size * grace
    if w > max_size or h > max_size:
        f = max(w / target_size, h / target_size)
        t = (max(1, round_half_up(w / f)), max(1, round_half_up(h / f)))
        img = resize(img, t, ResizeFilter.BOX)

    image_format = "png" if c > 3 or lossless else "jpg"

    _, encoded_img = cv2.imencode(f".{image_format}", to_uint8(img, normalized=True))  # type: ignore
    base64_img = base64.b64encode(encoded_img).decode("utf8")  # type: ignore

    return f"data:image/{image_format};base64,{base64_img}", img


class LargeImageOutput(ImageOutput):
    def __init__(
        self,
        label: str = "Image",
        image_type: navi.ExpressionJson = "Image",
        kind: OutputKind = "large-image",
        has_handle: bool = True,
        assume_normalized: bool = False,
    ) -> None:
        super().__init__(
            label,
            image_type=image_type,
            kind=kind,
            has_handle=has_handle,
            assume_normalized=assume_normalized,
        )

    def get_broadcast_data(self, value: np.ndarray):
        img = value
        h, w, c = get_h_w_c(img)
        image_size = max(h, w)

        preview_sizes = [2048, 1024, 512, 256]
        preview_size_grace = 1.2

        start_index = len(preview_sizes) - 1
        for i, size in enumerate(preview_sizes):
            if size <= image_size and image_size <= size * preview_size_grace:
                # this preview size will perfectly fit the image
                start_index = i
                break
            if image_size > size:
                # the image size is larger than the preview size, so try to pick the previous size
                start_index = max(0, i - 1)
                break

        previews = []

        # Encode for multiple scales. Use the preceding scale to save time encoding the smaller sizes.
        last_encoded = img
        for size in preview_sizes[start_index:]:
            largest_preview = size == preview_sizes[start_index]
            url, last_encoded = preview_encode(
                last_encoded,
                target_size=size,
                grace=preview_size_grace,
                lossless=largest_preview,
            )
            le_h, le_w, _ = get_h_w_c(last_encoded)
            previews.append({"width": le_w, "height": le_h, "url": url})

        return {
            "previews": previews,
            "height": h,
            "width": w,
            "channels": c,
        }


def VideoOutput():
    """Output a 3D Video NumPy array"""
    return NumPyOutput("Video", "Video")

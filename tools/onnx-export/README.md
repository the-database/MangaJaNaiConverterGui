# ONNX models for MangaJaNaiConverterGui 2.0

MangaJaNaiConverterGui 2.0 runs ONNX models through [animejanai-inference](https://github.com/the-database/animejanai-inference)
on TensorRT (NVIDIA) or DirectML (any DirectX 12 GPU). PyTorch models (`.pth`, `.safetensors`) can't
be used directly; export them to ONNX first.

## Converting your own model

Use [traiNNer-redux](https://github.com/the-database/traiNNer-redux)'s `convert_to_onnx.py`:

1. Install traiNNer-redux with its ONNX extras (`pip install .[onnx]`).
2. Copy the template for your model's architecture from `options/_templates/onnx/<ARCH>/` and set
   `network_g` to match the checkpoint (architecture and size, e.g. `esrgan` with
   `use_pixel_unshuffle: true` for 2x ESRGAN) and `path.pretrain_network_g` to the model file.
3. Use these export settings, which run on both TensorRT and DirectML:

   ```yaml
   onnx:
     dynamo: false   # the legacy exporter; DirectML needs opset 19 or lower
     dtype: fp16     # fp32 if the fp16 output is broken (NaN or visibly wrong)
     opset: 17
     shape: 1x3xHxW  # dynamic height and width
     verify: true
     optimize: false
   ```

4. Run `python convert_to_onnx.py -opt <your yml>` and copy the `.onnx` from `onnx/` into
   MangaJaNaiConverterGui's models folder (App Settings → Open Models Directory). Pick it in a
   workflow chain like any other model.

Notes:
- Don't export bf16: DirectML can't run it.
- Some architectures only accept tile sizes that are a multiple of 8 or 16 (FDAT, HAT). The tile
  sizes MangaJaNaiConverterGui offers all are.
- A model's first use on TensorRT builds an engine for your GPU, which takes a few minutes once
  per model and tile size.

## The default models

`configs/` holds the traiNNer-redux options used to export the default models (checkpoint paths
replaced with `path/to/`). They ship as two component packs, `models-mangajanai` and
`models-illustrationjanai`, built from the `models-*.7z` archives on a MangaJaNai release.

| Models | ONNX dtype | Notes |
|---|---|---|
| MangaJaNai V1 ESRGAN 2x/4x (14) | fp16 | Matches 1.x, whose default was fp16. fp32 would be no more accurate on TensorRT, which builds fp32 models as TF32 by default, and is ~3x slower. |
| IllustrationJaNai V3 FDAT_M, FDAT_XL, SPAN_S (8) | fp16 | 56–72 dB vs fp32 PyTorch on TensorRT and DirectML |
| IllustrationJaNai V3 DAT2 (2), HAT_L (1) | fp32 | fp16 overflows (activations above 65504), producing NaN |

All are opset 17 with input `input` and output `output` in NCHW RGB [0, 1], batch 1, dynamic
height/width. Gray pages are fed to them as one channel; the inference engine replicates it.

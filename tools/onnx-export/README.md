# ONNX models for MangaJaNai GUI 2.0

MangaJaNai GUI 2.0 runs ONNX models through [animejanai-inference](https://github.com/the-database/animejanai-inference)
on TensorRT-RTX or TensorRT (NVIDIA) or DirectML (any DirectX 12 GPU). PyTorch models (`.pth`,
`.safetensors`) can't be used directly; export them to ONNX first.

## Converting your own model

Use [traiNNer-redux](https://github.com/the-database/traiNNer-redux)'s `convert_to_onnx.py`:

1. Install traiNNer-redux with its ONNX extras (`pip install .[onnx]`).
2. Copy the template for your model's architecture from `options/_templates/onnx/<ARCH>/` and set
   `network_g` to match the checkpoint (architecture and size, e.g. `esrgan` with
   `use_pixel_unshuffle: true` for 2x ESRGAN) and `path.pretrain_network_g` to the model file.
3. Pick the precision for the GPU and backend you'll run it on (fastest first):

   | Runs on | Export | Notes |
   |---|---|---|
   | NVIDIA RTX 30-series and newer (TensorRT-RTX, TensorRT) | `bf16` or `fp16` | bf16 has fp32's range, so it suits models that overflow in fp16 (DAT, HAT and other transformers often do). |
   | NVIDIA RTX 20 / GTX 16 series | `fp16` | These GPUs have no bf16. Use `fp32` if the fp16 output is broken. |
   | DirectML (AMD, Intel, any NVIDIA) | `fp16` | DirectML can't run bf16. Use `fp32` if the fp16 output is broken. |

   An fp32 export runs on everything but is the slowest, typically by about 1.5-2x on TensorRT.
   Settings that run everywhere the precision allows:

   ```yaml
   onnx:
     dynamo: false   # the legacy exporter; DirectML needs opset 19 or lower
     dtype: fp16     # or bf16 / fp32, from the table above
     opset: 17
     shape: 1x3xHxW  # dynamic height and width
     verify: true
     optimize: false
   ```

4. Run `python convert_to_onnx.py -opt <your yml>` and copy the `.onnx` from `onnx/` into
   MangaJaNai GUI's models folder (App Settings → Open Models Directory). Pick it in a
   workflow chain like any other model.

Notes:
- Name the file `<model>_<precision>.onnx` (e.g. `4x_MyModel_fp16.onnx`), like the default models:
  the name then says what runs.
- Each chain shows the model's precision. If it's slower than it could be on your GPU and backend
  (an fp32 model where fp16 or bf16 would run), the tag's tooltip and the upscale log say so.
- Some architectures only accept tile sizes that are a multiple of 8 or 16 (FDAT, HAT). The tile
  sizes MangaJaNai GUI offers all are.
- A model's first use on TensorRT builds an engine for your GPU, which takes a few minutes once
  per model and tile size (TensorRT-RTX: seconds).

## The default models

`configs/` holds the traiNNer-redux options used to export the default models (checkpoint paths
replaced with `path/to/`). Every file is named `<checkpoint>_<precision of the ONNX>`; the
checkpoint names themselves end in the precision they were *trained* in, which the ONNX name
replaces. The exports are renamed to that name (traiNNer-redux appends the shape, dtype and opset).

| Models | ONNX precision | Pack | Notes |
|---|---|---|---|
| MangaJaNai V1 ESRGAN 2x/4x (14) | fp16 | `models-mangajanai` | Matches 1.x, whose default was fp16. On a real halftone page, 98.3% of pixels match fp32 (64.9 dB) at about half the time. |
| IllustrationJaNai V3 FDAT_M, FDAT_XL, SPAN_S (8) | fp16 | `models-illustrationjanai` | 56–72 dB vs fp32 PyTorch on TensorRT and DirectML. |
| IllustrationJaNai V3 DAT2 (2), HAT_L (1) | bf16 | `models-illustrationjanai-bf16` | Installed on NVIDIA GPUs with compute capability 8.0+ (RTX 30-series and newer). fp16 overflows in these, so bf16; 40-50% faster than fp32 on TensorRT and TensorRT-RTX, 52.5–53.7 dB vs fp32. |
| IllustrationJaNai V3 DAT2 (2), HAT_L (1) | fp32 | `models-illustrationjanai-fp32` | Installed everywhere else (DirectML, RTX 20 / GTX 16), which can't run bf16. |

The fp16 and fp32 exports are opset 17 (legacy exporter); the bf16 ones are DAT2 at opset 23
(dynamo exporter) and HAT_L at opset 22. All take input `input` and give output `output` in NCHW
RGB [0, 1], batch 1, dynamic height/width, as fp32 tensors whatever the precision inside. Gray
pages are fed to them as one channel; the inference engine replicates it.

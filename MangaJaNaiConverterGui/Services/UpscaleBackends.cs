using System.Collections.Generic;

namespace MangaJaNaiConverterGui.Services
{
    // The App Settings backend choices (the saved UpscaleBackend strings) and what each one asks of
    // mangajanai_upscale. A saved value this version doesn't know counts as Auto.
    public static class UpscaleBackends
    {
        public const string Auto = "Auto";
        public const string TensorRt = "TensorRT";
        // Experimental and opt-in (the trt-rtx component): only ever an explicit choice.
        public const string TensorRtRtx = "TensorRT-RTX";
        public const string DirectMl = "DirectML";

        // mangajanai_upscale's --backend names.
        public const string CliAuto = "auto";
        public const string CliTensorRt = "tensorrt";
        public const string CliTensorRtRtx = "tensorrt-rtx";
        public const string CliDirectMl = "directml";

        public static string Normalize(string? choice) =>
            choice is TensorRt or TensorRtRtx or DirectMl ? choice : Auto;

        // The backends asked for --list-devices, in order, until one lists a device. Auto: TensorRT
        // where it can run, else DirectML; it never picks TensorRT-RTX. TensorRT-RTX: its own list,
        // else TensorRT's (the same cards, under the same PCI-address ids).
        public static string[] DeviceQueryOrder(string? choice) => Normalize(choice) switch
        {
            TensorRt => [CliTensorRt],
            TensorRtRtx => [CliTensorRtRtx, CliTensorRt],
            DirectMl => [CliDirectMl],
            _ => [CliTensorRt, CliDirectMl],
        };

        // The --backend an upscale runs with, given the backend whose list supplied the devices (null
        // when none listed any). TensorRT-RTX stays TensorRT-RTX on TensorRT's device list: the CLI
        // falls back to TensorRT on its own when TensorRT-RTX can't start.
        public static string UpscaleBackend(string? choice, string? listedBy) =>
            listedBy is null ? CliAuto
            : Normalize(choice) == TensorRtRtx ? CliTensorRtRtx
            : listedBy;

        public static string DisplayName(string cliBackend) => cliBackend switch
        {
            CliTensorRt => TensorRt,
            CliTensorRtRtx => TensorRtRtx,
            CliDirectMl => DirectMl,
            _ => Auto,
        };

        // TensorRT-RTX runs on the trt-rtx component and needs trt-runtime too (its CUDA runtime, and
        // TensorRT for the fallback). Names what's missing, as App Settings > Components lists it;
        // null when the choice isn't TensorRT-RTX or nothing is missing.
        public static string? MissingComponentsMessage(string? choice, bool trtRtxInstalled, bool trtRuntimeInstalled)
        {
            if (Normalize(choice) != TensorRtRtx || (trtRtxInstalled && trtRuntimeInstalled))
            {
                return null;
            }
            var missing = new List<string>();
            if (!trtRtxInstalled)
            {
                missing.Add("\"TensorRT-RTX (experimental)\"");
            }
            if (!trtRuntimeInstalled)
            {
                missing.Add("\"TensorRT runtime\"");
            }
            return missing.Count == 1
                ? $"TensorRT-RTX needs the {missing[0]} component. Install it in App Settings > Components, or choose another backend."
                : $"TensorRT-RTX needs the {missing[0]} and {missing[1]} components. Install them in App Settings > Components, or choose another backend.";
        }
    }
}

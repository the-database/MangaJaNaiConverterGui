using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace MangaJaNaiConverterGui.Services
{
    // The shipped models, by family. A shipped file's name is its checkpoint's base name plus the
    // ONNX file's actual precision ("..._DAT2_28k_bf16.onnx"); a family is that base name, and some
    // ship in more than one precision. A chain naming a shipped family moves to the best variant
    // installed (Resolve), visibly: the chain's model changes, so the list shows what runs. Custom
    // models (any other name) are never touched.
    public static class ModelVariants
    {
        public const string Fp16 = "fp16";
        public const string Bf16 = "bf16";
        public const string Fp32 = "fp32";

        public const string MangaJaNaiPack = "models-mangajanai";
        public const string IllustrationJaNaiPack = "models-illustrationjanai";
        public const string IllustrationJaNaiBf16Pack = "models-illustrationjanai-bf16";
        public const string IllustrationJaNaiFp32Pack = "models-illustrationjanai-fp32";

        // A shipped family: its base name and, per precision it ships in, the component pack holding it.
        public sealed record Family(string Base, IReadOnlyDictionary<string, string> Packs);

        private static readonly Dictionary<string, Family> Families = BuildFamilies();

        private static Dictionary<string, Family> BuildFamilies()
        {
            var families = new Dictionary<string, Family>(StringComparer.OrdinalIgnoreCase);
            void Add(string baseName, params (string Precision, string Pack)[] variants) =>
                families.Add(baseName, new Family(baseName, variants.ToDictionary(v => v.Precision, v => v.Pack)));

            foreach (var b in new[]
            {
                "2x_MangaJaNai_1200p_V1_ESRGAN_70k", "2x_MangaJaNai_1300p_V1_ESRGAN_75k",
                "2x_MangaJaNai_1400p_V1_ESRGAN_70k", "2x_MangaJaNai_1500p_V1_ESRGAN_90k",
                "2x_MangaJaNai_1600p_V1_ESRGAN_90k", "2x_MangaJaNai_1920p_V1_ESRGAN_70k",
                "2x_MangaJaNai_2048p_V1_ESRGAN_95k",
                "4x_MangaJaNai_1200p_V1_ESRGAN_70k", "4x_MangaJaNai_1300p_V1_ESRGAN_75k",
                "4x_MangaJaNai_1400p_V1_ESRGAN_105k", "4x_MangaJaNai_1500p_V1_ESRGAN_105k",
                "4x_MangaJaNai_1600p_V1_ESRGAN_70k", "4x_MangaJaNai_1920p_V1_ESRGAN_105k",
                "4x_MangaJaNai_2048p_V1_ESRGAN_70k",
            })
            {
                Add(b, (Fp16, MangaJaNaiPack));
            }
            foreach (var b in new[]
            {
                "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k", "2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k",
                "2x_IllustrationJaNai_V3denoise_SPAN_S_30k", "2x_IllustrationJaNai_V3detail_SPAN_S_40k",
                "4x_IllustrationJaNai_V3denoise_FDAT_M_47k", "4x_IllustrationJaNai_V3detail_FDAT_M_40k",
                "4x_IllustrationJaNai_V3denoise_FDAT_XL_32k", "4x_IllustrationJaNai_V3detail_FDAT_XL_27k",
            })
            {
                Add(b, (Fp16, IllustrationJaNaiPack));
            }
            foreach (var b in new[]
            {
                "4x_IllustrationJaNai_V3denoise_DAT2_27k", "4x_IllustrationJaNai_V3detail_DAT2_28k",
                "4x_IllustrationJaNai_V3detail_HAT_L_28k",
            })
            {
                Add(b, (Bf16, IllustrationJaNaiBf16Pack), (Fp32, IllustrationJaNaiFp32Pack));
            }
            return families;
        }

        // Every shipped model file name.
        public static IEnumerable<string> ShippedModels =>
            Families.Values.SelectMany(f => f.Packs.Keys.Select(p => $"{f.Base}_{p}.onnx"));

        // "4x_..._DAT2_28k_bf16" -> ("4x_..._DAT2_28k", "bf16"); a stem without a precision suffix
        // comes back whole, with a null precision.
        public static (string Base, string? Precision) Split(string stem)
        {
            var m = Regex.Match(stem, "^(.+)_(fp16|bf16|fp32)$", RegexOptions.IgnoreCase);
            return m.Success ? (m.Groups[1].Value, m.Groups[2].Value.ToLowerInvariant()) : (stem, null);
        }

        // The precision a model's name states ("fp16", "bf16", "fp32"), or null.
        public static string? NamedPrecision(string? model) =>
            string.IsNullOrEmpty(model) ? null : Split(Path.GetFileNameWithoutExtension(model)).Precision;

        // The shipped family an ONNX model name belongs to: its name without the precision suffix
        // (or the whole name, for the 2.0 prerelease's unsuffixed MangaJaNai files) is a shipped base.
        // Null for a custom model, a path, or a non-ONNX file.
        public static Family? FindFamily(string? model)
        {
            if (string.IsNullOrWhiteSpace(model) || !model.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
                Path.GetFileName(model) != model)
            {
                return null;
            }
            return Families.TryGetValue(Split(Path.GetFileNameWithoutExtension(model)).Base, out var family) ? family : null;
        }

        // 1.x's default settings pointed every Custom Workflow's color chain at IllustrationJaNai V1,
        // which 2.0 doesn't ship (and 1.x didn't download either): such a chain moves to the 4x
        // IllustrationJaNai model the default workflow uses.
        private const string LegacyIllustrationModel = "4x_IllustrationJaNai_V1_ESRGAN_135k";
        private const string LegacyIllustrationReplacement = "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx";

        // 1.x's default models were PyTorch files (.pth / .safetensors); 2.0 ships their ONNX exports
        // under the same checkpoint names, so a saved or imported chain moves to the .onnx name (which
        // Resolve then matches to the installed variant). Only the name: needs no installed models.
        public static string MigrateLegacyName(string model)
        {
            if (string.IsNullOrWhiteSpace(model) || model.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                return model;
            }
            var ext = Path.GetExtension(model);
            var pytorch = ext.Equals(".pth", StringComparison.OrdinalIgnoreCase) || ext.Equals(".safetensors", StringComparison.OrdinalIgnoreCase);
            if (pytorch && Path.GetFileNameWithoutExtension(model).Equals(LegacyIllustrationModel, StringComparison.OrdinalIgnoreCase))
            {
                return LegacyIllustrationReplacement;
            }
            return pytorch && Regex.IsMatch(Path.GetFileName(model), @"^\dx_(MangaJaNai|IllustrationJaNai)_", RegexOptions.IgnoreCase)
                ? Path.ChangeExtension(model, ".onnx")
                : model;
        }

        // What a chain should run: Model, and when nothing installed can run there, the component
        // pack that has a variant that can (MissingPack, else null).
        public readonly record struct Choice(string Model, string? MissingPack);

        // The variant rule. For a shipped family's name (after the 1.x name migration):
        // - the named file is installed: keep it, except a bf16 file on DirectML, which can't run it;
        // - else the installed variant of its family, preferring bf16, fp16, fp32 where bf16 runs
        //   (TensorRT or TensorRT-RTX on compute capability 8.0+), else fp16, fp32 (never bf16);
        // - else keep the name; on DirectML a bf16 one then names the pack to install.
        // Custom models, "No Model" and empty names come back as they are (after the 1.x migration).
        public static Choice Resolve(string saved, IEnumerable<string> installed, bool bf16Runnable, bool directMl)
        {
            var model = MigrateLegacyName(saved ?? string.Empty);
            var family = FindFamily(model);
            if (family is null)
            {
                return new Choice(model, null);
            }

            var files = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var f in installed)
            {
                files.TryAdd(f, f);
            }

            var bf16Blocked = directMl && NamedPrecision(model) == Bf16;
            if (!bf16Blocked && files.TryGetValue(model, out var named))
            {
                return new Choice(named, null);
            }
            string[] order = bf16Runnable && !directMl ? [Bf16, Fp16, Fp32] : [Fp16, Fp32];
            foreach (var precision in order)
            {
                if (files.TryGetValue($"{family.Base}_{precision}.onnx", out var sibling))
                {
                    return new Choice(sibling, null);
                }
            }
            string? missingPack = null;
            if (bf16Blocked)
            {
                missingPack = family.Packs.TryGetValue(Fp16, out var fp16Pack) ? fp16Pack
                    : family.Packs.TryGetValue(Fp32, out var fp32Pack) ? fp32Pack : null;
            }
            return new Choice(model, missingPack);
        }

        // Whether bf16 models run: TensorRT or TensorRT-RTX (mangajanai_upscale's --backend names) on
        // a GPU of compute capability 8.0 or newer (RTX 30-series and newer). DirectML and Turing
        // can't; an unknown backend or GPU counts as can't.
        public static bool Bf16Runnable(string? cliBackend, string? computeCapability) =>
            (cliBackend is UpscaleBackends.CliTensorRt or UpscaleBackends.CliTensorRtRtx) &&
            ComputeCapabilityMajor(computeCapability) >= 8;

        // "8.9" -> 8, "12.0" -> 12; null for null or anything else.
        public static int? ComputeCapabilityMajor(string? computeCapability)
        {
            if (string.IsNullOrWhiteSpace(computeCapability))
            {
                return null;
            }
            var major = computeCapability.Split('.')[0];
            return int.TryParse(major, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out var m) ? m : null;
        }
    }
}

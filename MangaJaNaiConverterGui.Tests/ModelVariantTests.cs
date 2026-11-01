using Avalonia.Collections;
using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using Newtonsoft.Json;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    // Shipped model names state the ONNX file's precision; a chain naming a shipped family runs the
    // best variant installed for the backend and GPU (ModelVariants.Resolve).
    public class ModelVariantTests
    {
        const string Dat2 = "4x_IllustrationJaNai_V3detail_DAT2_28k";
        const string Dat2Bf16 = Dat2 + "_bf16.onnx";
        const string Dat2Fp32 = Dat2 + "_fp32.onnx";
        const string Manga = "2x_MangaJaNai_1200p_V1_ESRGAN_70k";
        const string MangaFp16 = Manga + "_fp16.onnx";
        const string FdatXl = "4x_IllustrationJaNai_V3denoise_FDAT_XL_32k";

        // The spec's shipped list, written out independently of ModelVariants.
        static readonly string[] Shipped =
        [
            "2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx", "2x_MangaJaNai_1300p_V1_ESRGAN_75k_fp16.onnx",
            "2x_MangaJaNai_1400p_V1_ESRGAN_70k_fp16.onnx", "2x_MangaJaNai_1500p_V1_ESRGAN_90k_fp16.onnx",
            "2x_MangaJaNai_1600p_V1_ESRGAN_90k_fp16.onnx", "2x_MangaJaNai_1920p_V1_ESRGAN_70k_fp16.onnx",
            "2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.onnx",
            "4x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx", "4x_MangaJaNai_1300p_V1_ESRGAN_75k_fp16.onnx",
            "4x_MangaJaNai_1400p_V1_ESRGAN_105k_fp16.onnx", "4x_MangaJaNai_1500p_V1_ESRGAN_105k_fp16.onnx",
            "4x_MangaJaNai_1600p_V1_ESRGAN_70k_fp16.onnx", "4x_MangaJaNai_1920p_V1_ESRGAN_105k_fp16.onnx",
            "4x_MangaJaNai_2048p_V1_ESRGAN_70k_fp16.onnx",
            "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx", "2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.onnx",
            "2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.onnx", "2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.onnx",
            "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx", "4x_IllustrationJaNai_V3detail_FDAT_M_40k_fp16.onnx",
            "4x_IllustrationJaNai_V3denoise_FDAT_XL_32k_fp16.onnx", "4x_IllustrationJaNai_V3detail_FDAT_XL_27k_fp16.onnx",
            "4x_IllustrationJaNai_V3denoise_DAT2_27k_bf16.onnx", "4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx",
            "4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx",
            "4x_IllustrationJaNai_V3denoise_DAT2_27k_fp32.onnx", "4x_IllustrationJaNai_V3detail_DAT2_28k_fp32.onnx",
            "4x_IllustrationJaNai_V3detail_HAT_L_28k_fp32.onnx",
        ];

        [Fact]
        public void TheShippedModelsAreTheSpecsList() =>
            Assert.Equal(Shipped.Order(), ModelVariants.ShippedModels.Order());

        static string Resolve(string saved, string[] installed, bool bf16Runnable = false, bool directMl = false) =>
            ModelVariants.Resolve(saved, installed, bf16Runnable, directMl).Model;

        // ---- families --------------------------------------------------------------------------

        [Theory]
        [InlineData(Dat2Bf16, Dat2)]
        [InlineData(Dat2Fp32, Dat2)]
        [InlineData(MangaFp16, Manga)]
        // the 2.0 prerelease's unsuffixed MangaJaNai names
        [InlineData(Manga + ".onnx", Manga)]
        // the prerelease's FDAT XL names carried the checkpoint's precision
        [InlineData(FdatXl + "_bf16.onnx", FdatXl)]
        // 1.x PyTorch names, through the 1.x migration
        [InlineData(Manga + ".pth", Manga)]
        [InlineData(Dat2 + "_bf16.safetensors", Dat2)]
        [InlineData("4x_IllustrationJaNai_V1_ESRGAN_135k.pth", "4x_IllustrationJaNai_V3denoise_FDAT_M_47k")]
        // case doesn't matter on Windows; the canonical base comes back
        [InlineData("4X_ILLUSTRATIONJANAI_V3DETAIL_DAT2_28K_FP32.ONNX", Dat2)]
        public void ShippedNamesBelongToTheirFamily(string model, string family)
        {
            var f = ModelVariants.FindFamily(ModelVariants.MigrateLegacyName(model));
            Assert.NotNull(f);
            Assert.Equal(family, f.Base);
        }

        [Theory]
        [InlineData("4x_MyModel_fp16.onnx")]
        [InlineData("4x_IllustrationJaNai_V1_ESRGAN_135k.onnx")]
        [InlineData("2x_MangaJaNai_1200p_V1_ESRGAN_70k_custom.onnx")]
        [InlineData(@"C:\models\2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx")]
        [InlineData("2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.pth.bak")]
        [InlineData("No Model")]
        [InlineData("")]
        public void OtherNamesAreCustom(string model) =>
            Assert.Null(ModelVariants.FindFamily(model));

        [Theory]
        [InlineData(Dat2Bf16, "bf16")]
        [InlineData(Dat2Fp32, "fp32")]
        [InlineData("x_FP16.onnx", "fp16")]
        [InlineData(Manga + ".onnx", null)]
        [InlineData("No Model", null)]
        public void NamedPrecision(string model, string? precision) =>
            Assert.Equal(precision, ModelVariants.NamedPrecision(model));

        // ---- the variant rule ------------------------------------------------------------------

        static readonly string[] BothDat2 = [Dat2Bf16, Dat2Fp32, MangaFp16, "No Model"];
        static readonly string[] OnlyFp32 = [Dat2Fp32, MangaFp16, "No Model"];
        static readonly string[] OnlyBf16 = [Dat2Bf16, MangaFp16, "No Model"];

        // The named file is installed: it stays, whatever the GPU.
        [Theory]
        [InlineData(Dat2Bf16, true)]
        [InlineData(Dat2Bf16, false)]
        [InlineData(Dat2Fp32, true)]
        [InlineData(Dat2Fp32, false)]
        public void AnInstalledModelStays(string saved, bool bf16Runnable) =>
            Assert.Equal(saved, Resolve(saved, BothDat2, bf16Runnable));

        // The old IllustrationJaNai _bf16 names were fp32 files; the same names are now the bf16
        // files. Where bf16 is installed they name it; elsewhere the fp32 file installed instead.
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void AnOldBf16NameMovesToTheFp32FileWhereOnlyItIsInstalled(bool bf16Runnable) =>
            Assert.Equal(Dat2Fp32, Resolve(Dat2Bf16, OnlyFp32, bf16Runnable));

        [Fact]
        public void AnOldBf16NameStaysWhereTheBf16FileIsInstalled() =>
            Assert.Equal(Dat2Bf16, Resolve(Dat2Bf16, OnlyBf16, bf16Runnable: true));

        // Missing: TensorRT / TensorRT-RTX on compute capability 8.0+ prefer bf16, fp16, fp32.
        [Fact]
        public void AMissingFp32ModelMovesToBf16WhereBf16Runs() =>
            Assert.Equal(Dat2Bf16, Resolve(Dat2Fp32, OnlyBf16, bf16Runnable: true));

        [Fact]
        public void Bf16IsPreferredOverFp16AndFp32() =>
            Assert.Equal(Dat2Bf16, Resolve(Dat2 + "_fp16.onnx", BothDat2, bf16Runnable: true));

        [Fact]
        public void Fp16IsPreferredOverFp32WhereBf16Runs() =>
            Assert.Equal(Dat2 + "_fp16.onnx", Resolve(Dat2 + ".onnx", [Dat2Fp32, Dat2 + "_fp16.onnx"], bf16Runnable: true));

        // Otherwise (DirectML, Turing or older, unknown): fp16, fp32, never bf16.
        [Theory]
        [InlineData(false, false)]
        [InlineData(false, true)]
        public void WithoutBf16AMissingModelNeverMovesToBf16(bool bf16Runnable, bool directMl) =>
            Assert.Equal(Dat2 + "_fp16.onnx", Resolve(Dat2 + "_fp32.onnx", [Dat2Bf16, Dat2 + "_fp16.onnx"], bf16Runnable, directMl));

        [Fact]
        public void WithoutBf16OnlyABf16SiblingLeavesTheChainAsItIs() =>
            Assert.Equal(Dat2Fp32, Resolve(Dat2Fp32, OnlyBf16, bf16Runnable: false));

        // Turing on TensorRT can't run bf16 either, but an installed file is the user's choice: it
        // stays (its precision tag carries the CLI's hint).
        [Fact]
        public void AnInstalledBf16ModelStaysOnTuring() =>
            Assert.Equal(Dat2Bf16, Resolve(Dat2Bf16, BothDat2, bf16Runnable: false, directMl: false));

        // DirectML can't run bf16 at all: an installed bf16 model moves to its fp32 sibling.
        [Fact]
        public void DirectMlMovesABf16ModelToItsFp32Sibling()
        {
            var choice = ModelVariants.Resolve(Dat2Bf16, BothDat2, false, true);
            Assert.Equal(Dat2Fp32, choice.Model);
            Assert.Null(choice.MissingPack);
        }

        // ...and without one, the chain keeps its name and the pack to install is named.
        [Fact]
        public void DirectMlWithoutTheFp32SiblingNamesTheFp32Pack()
        {
            var choice = ModelVariants.Resolve(Dat2Bf16, OnlyBf16, false, true);
            Assert.Equal(Dat2Bf16, choice.Model);
            Assert.Equal(ModelVariants.IllustrationJaNaiFp32Pack, choice.MissingPack);
        }

        // The prerelease's FDAT XL "_bf16" was an fp16 file: on DirectML it moves to the fp16 one,
        // and without it names the pack that has it.
        [Fact]
        public void DirectMlMovesAnOldFdatXlNameToFp16()
        {
            Assert.Equal(FdatXl + "_fp16.onnx", Resolve(FdatXl + "_bf16.onnx", [FdatXl + "_fp16.onnx"], directMl: true));
            Assert.Equal(ModelVariants.IllustrationJaNaiPack, ModelVariants.Resolve(FdatXl + "_bf16.onnx", [], false, true).MissingPack);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void AnOldFdatXlNameMovesToFp16(bool bf16Runnable) =>
            Assert.Equal(FdatXl + "_fp16.onnx", Resolve(FdatXl + "_bf16.onnx", [FdatXl + "_fp16.onnx"], bf16Runnable));

        [Fact]
        public void NoProblemIsReportedOffDirectMl() =>
            Assert.Null(ModelVariants.Resolve(Dat2Bf16, [], true, false).MissingPack);

        // Nothing of the family installed: the name stays (the model is just missing).
        [Fact]
        public void NothingInstalledLeavesTheName() =>
            Assert.Equal(Dat2Fp32, Resolve(Dat2Fp32, [MangaFp16], bf16Runnable: true));

        // ---- legacy names ------------------------------------------------------------------------

        [Theory]
        // 2.0 prerelease: unsuffixed MangaJaNai
        [InlineData(Manga + ".onnx", MangaFp16)]
        // 1.x PyTorch
        [InlineData(Manga + ".pth", MangaFp16)]
        [InlineData(Manga + ".safetensors", MangaFp16)]
        [InlineData("4x_IllustrationJaNai_V1_ESRGAN_135k.safetensors", "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx")]
        [InlineData("4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.pth", "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx")]
        // case: the installed file's spelling
        [InlineData("2X_MANGAJANAI_1200P_V1_ESRGAN_70K_FP16.ONNX", MangaFp16)]
        public void LegacyNamesMoveToTheInstalledFile(string saved, string model) =>
            Assert.Equal(model, Resolve(saved, [.. Shipped, "No Model"], bf16Runnable: true));

        [Theory]
        [InlineData(Dat2 + "_bf16.pth", true, Dat2Bf16)]
        [InlineData(Dat2 + "_bf16.pth", false, Dat2Fp32)]
        public void A1xDat2ModelGetsTheVariantForTheGpu(string saved, bool bf16Runnable, string model) =>
            Assert.Equal(model, Resolve(saved, bf16Runnable ? OnlyBf16 : OnlyFp32, bf16Runnable));

        [Theory]
        [InlineData("4x_MyModel_fp32.onnx")]
        [InlineData("4x_MyModel_bf16.onnx")]
        [InlineData("4x_MyModel.pth")]
        [InlineData("4x_IllustrationJaNai_V1_ESRGAN_135k.onnx")]
        [InlineData("No Model")]
        [InlineData("")]
        public void CustomModelsAreNeverTouched(string saved)
        {
            foreach (var (bf16, dml) in new[] { (true, false), (false, false), (false, true) })
            {
                var choice = ModelVariants.Resolve(saved, [.. Shipped, "4x_MyModel_fp16.onnx"], bf16, dml);
                Assert.Equal(saved, choice.Model);
                Assert.Null(choice.MissingPack);
            }
        }

        // ---- bf16 runnable -------------------------------------------------------------------------

        [Theory]
        [InlineData("tensorrt", "8.9", true)]
        [InlineData("tensorrt-rtx", "8.0", true)]
        [InlineData("tensorrt-rtx", "12.0", true)]
        [InlineData("tensorrt", "7.5", false)]
        [InlineData("tensorrt-rtx", "7.5", false)]
        [InlineData("directml", "8.9", false)]
        [InlineData("directml", null, false)]
        [InlineData("tensorrt", null, false)]
        [InlineData("tensorrt", "garbage", false)]
        [InlineData("auto", "8.9", false)]
        public void Bf16Runnable(string backend, string? computeCapability, bool runnable) =>
            Assert.Equal(runnable, ModelVariants.Bf16Runnable(backend, computeCapability));

        // ---- the shipped default settings ---------------------------------------------------------

        [Fact]
        public void TheDefaultStateNamesOnlyShippedModels()
        {
            var names = Regex.Matches(File.ReadAllText(NewtonsoftJsonSuspensionDriver.DefaultStatePath), "\"ModelFilePath\": \"([^\"]*)\"")
                .Select(m => m.Groups[1].Value).ToList();
            Assert.NotEmpty(names);
            Assert.All(names, n => Assert.Contains(n, Shipped));
        }

        [Fact]
        public void TheResetWorkflowNamesOnlyShippedModels()
        {
            var json = (string)typeof(MainWindowViewModel).GetField("DEFAULT_WORKFLOW", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;
            var workflow = JsonConvert.DeserializeObject<UpscaleWorkflow>(json, NewtonsoftJsonSuspensionDriver.Settings)!;
            Assert.All(workflow.Chains, c => Assert.Contains(c.ModelFilePath, Shipped));
        }

        // ---- applied to the view model's chains -----------------------------------------------------

        static (MainWindowViewModel Vm, UpscaleChain Chain) VmWithChain(string model, params string[] installed)
        {
            var backend = new FakeBackendService(installed);
            var chain = new UpscaleChain(backend) { ChainNumber = "1", ModelFilePath = model };
            var workflow = new UpscaleWorkflow { WorkflowName = "Test", Chains = [chain] };
            var vm = new MainWindowViewModel(backend) { Workflows = [workflow] };
            workflow.Vm = vm;
            chain.Vm = vm;
            return (vm, chain);
        }

        // Switching to DirectML moves a bf16 chain to fp32; switching back returns it to bf16 (the
        // installed bf16 file is the chain's choice), and a model picked meanwhile is kept.
        [Fact]
        public void ADirectMlMoveUndoesItselfWhenTheBackendChangesBack()
        {
            var (vm, chain) = VmWithChain(Dat2Bf16, BothDat2);

            vm.UpscaleBackend = UpscaleBackends.DirectMl;
            vm.ApplyModelVariants();
            Assert.Equal(Dat2Fp32, chain.ModelFilePath);

            vm.UpscaleBackend = UpscaleBackends.Auto;
            vm.ApplyModelVariants();
            Assert.Equal(Dat2Bf16, chain.ModelFilePath);
            Assert.Null(chain.VariantSource);
        }

        [Fact]
        public void AModelPickedAfterAMoveIsKept()
        {
            var (vm, chain) = VmWithChain(Dat2Bf16, BothDat2);
            vm.UpscaleBackend = UpscaleBackends.DirectMl;
            vm.ApplyModelVariants();

            chain.ModelFilePath = Dat2Fp32; // picked (same file, but now the user's choice)
            chain.ModelFilePath = MangaFp16;
            vm.UpscaleBackend = UpscaleBackends.Auto;
            vm.ApplyModelVariants();
            Assert.Equal(MangaFp16, chain.ModelFilePath);
        }

        [Fact]
        public void DirectMlWithoutTheFp32PackIsAValidationProblem()
        {
            var (vm, chain) = VmWithChain(Dat2Bf16, OnlyBf16);
            vm.UpscaleBackend = UpscaleBackends.DirectMl;
            vm.ApplyModelVariants();

            Assert.Equal(Dat2Bf16, chain.ModelFilePath);
            Assert.Equal("Chain 1's model 4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx is bf16, which DirectML can't run. Install the \"IllustrationJaNai DAT2/HAT, fp32 (DirectML, older NVIDIA)\" component in App Settings > Components, or choose another backend.",
                vm.ModelVariantProblem(chain));
        }

        [Fact]
        public void ABf16ModelPickedOnDirectMlAsksForItsFp32Sibling()
        {
            var (vm, chain) = VmWithChain(MangaFp16, BothDat2);
            vm.UpscaleBackend = UpscaleBackends.DirectMl;
            vm.ApplyModelVariants();
            chain.ModelFilePath = Dat2Bf16; // picked by hand: not moved behind the user's back

            Assert.Equal(Dat2Bf16, chain.ModelFilePath);
            Assert.Equal("Chain 1's model 4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx is bf16, which DirectML can't run. Select 4x_IllustrationJaNai_V3detail_DAT2_28k_fp32.onnx instead.",
                vm.ModelVariantProblem(chain));
        }

        [Fact]
        public void NoProblemOffDirectMl()
        {
            var (vm, chain) = VmWithChain(Dat2Bf16, OnlyBf16);
            vm.ApplyModelVariants();
            Assert.Null(vm.ModelVariantProblem(chain));
        }

        // The tag and the variant bookkeeping aren't saved: the settings format stays as it was.
        [Fact]
        public void TheChainSavesNoNewSettings()
        {
            var (vm, chain) = VmWithChain(Dat2Bf16, BothDat2);
            vm.UpscaleBackend = UpscaleBackends.DirectMl;
            vm.ApplyModelVariants();

            var json = JsonConvert.SerializeObject(chain, NewtonsoftJsonSuspensionDriver.Settings);
            Assert.DoesNotContain("Precision", json);
            Assert.DoesNotContain("Variant", json);
            Assert.Contains($"\"ModelFilePath\": \"{Dat2Fp32}\"", json);
        }

        [Fact]
        public void TheViewModelSavesNoNewSettings()
        {
            var (vm, _) = VmWithChain(Dat2Bf16, BothDat2);
            vm.ModelHintText = "x.onnx: hint";
            vm.BackendSetupHint = "TensorRT-RTX is much faster on X.";

            var json = JsonConvert.SerializeObject(vm, NewtonsoftJsonSuspensionDriver.Settings);
            foreach (var name in new[] { "ModelHint", "BackendSetupHint", "Bf16Runnable", "BackendIsDirectMl", "Precision", "Variant" })
            {
                Assert.DoesNotContain(name, json);
            }
        }

        sealed class FakeBackendService(string[] models) : IBackendService
        {
            public string AppStateFolder => Path.GetTempPath();
            public string AppStateFilename => "appstate2.json";
            public string AppStatePath => Path.Combine(AppStateFolder, AppStateFilename);
            public string LogsDirectory => AppStateFolder;
            public string InferenceDirectory => Path.Combine(AppStateFolder, "no-such-inference-dir");
            public string BackendExePath => Path.Combine(InferenceDirectory, "mangajanai_upscale.exe");
            public bool IsBackendInstalled => false;
            public bool IsTensorRtRtxInstalled => false;
            public bool IsTensorRtRuntimeInstalled => false;
            public string UserModelsDirectory => AppStateFolder;
            public string BundledModelsDirectory => AppStateFolder;
            public bool AreModelsInstalled => true;
            public string EngineCacheDirectory => AppStateFolder;
            public AvaloniaList<string> AllModels { get; } = new(models);
            public bool RefreshModels() => false;
            public string? ResolveModelPath(string? model) => null;
        }
    }
}

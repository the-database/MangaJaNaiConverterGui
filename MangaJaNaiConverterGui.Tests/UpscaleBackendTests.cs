using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.IO;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    // The App Settings backend choice -> what mangajanai_upscale is asked for.
    public class UpscaleBackendTests
    {
        [Theory]
        [InlineData("Auto", false, new[] { "tensorrt", "directml" })]
        [InlineData("TensorRT", false, new[] { "tensorrt" })]
        [InlineData("TensorRT-RTX", false, new[] { "tensorrt-rtx", "tensorrt" })]
        [InlineData("DirectML", false, new[] { "directml" })]
        // A value from a newer version (or garbage) is Auto.
        [InlineData("SomethingNew", false, new[] { "tensorrt", "directml" })]
        [InlineData(null, false, new[] { "tensorrt", "directml" })]
        // With the TensorRT-RTX component installed, Auto tries it first; explicit choices don't change.
        [InlineData("Auto", true, new[] { "tensorrt-rtx", "tensorrt", "directml" })]
        [InlineData(null, true, new[] { "tensorrt-rtx", "tensorrt", "directml" })]
        [InlineData("TensorRT", true, new[] { "tensorrt" })]
        [InlineData("TensorRT-RTX", true, new[] { "tensorrt-rtx", "tensorrt" })]
        [InlineData("DirectML", true, new[] { "directml" })]
        public void DeviceListQueries(string? choice, bool trtRtxInstalled, string[] backends) =>
            Assert.Equal(backends, UpscaleBackends.DeviceQueryOrder(choice, trtRtxInstalled));

        [Theory]
        [InlineData("Auto", "tensorrt-rtx", "tensorrt-rtx")]
        [InlineData("Auto", "tensorrt", "tensorrt")]
        [InlineData("Auto", "directml", "directml")]
        [InlineData("TensorRT", "tensorrt", "tensorrt")]
        [InlineData("DirectML", "directml", "directml")]
        [InlineData("TensorRT-RTX", "tensorrt-rtx", "tensorrt-rtx")]
        // TensorRT-RTX listed nothing, so the devices came from TensorRT: still a TensorRT-RTX upscale
        // (the CLI falls back to TensorRT itself).
        [InlineData("TensorRT-RTX", "tensorrt", "tensorrt-rtx")]
        // No device found at all: the CLI picks.
        [InlineData("TensorRT-RTX", null, "auto")]
        [InlineData("TensorRT", null, "auto")]
        [InlineData("SomethingNew", "tensorrt", "tensorrt")]
        public void UpscaleBackend(string choice, string? listedBy, string cli) =>
            Assert.Equal(cli, UpscaleBackends.UpscaleBackend(choice, listedBy));

        [Theory]
        [InlineData("tensorrt", "TensorRT")]
        [InlineData("tensorrt-rtx", "TensorRT-RTX")]
        [InlineData("directml", "DirectML")]
        public void DisplayNames(string cli, string name) =>
            Assert.Equal(name, UpscaleBackends.DisplayName(cli));

        [Fact]
        public void OtherBackendsNeedNoTensorRtRtxComponents()
        {
            Assert.Null(UpscaleBackends.MissingComponentsMessage("Auto", false, false));
            Assert.Null(UpscaleBackends.MissingComponentsMessage("TensorRT", false, false));
            Assert.Null(UpscaleBackends.MissingComponentsMessage("DirectML", false, false));
            Assert.Null(UpscaleBackends.MissingComponentsMessage("TensorRT-RTX", true, true));
        }

        [Fact]
        public void MissingTensorRtRtxNamesTheComponentToInstall()
        {
            var message = UpscaleBackends.MissingComponentsMessage("TensorRT-RTX", false, true);
            Assert.Equal("TensorRT-RTX needs the \"TensorRT-RTX\" component. Install it in App Settings > Components, or choose another backend.", message);
        }

        [Fact]
        public void MissingTensorRtRuntimeNamesTheComponentToInstall()
        {
            var message = UpscaleBackends.MissingComponentsMessage("TensorRT-RTX", true, false);
            Assert.Equal("TensorRT-RTX needs the \"TensorRT runtime\" component. Install it in App Settings > Components, or choose another backend.", message);
        }

        [Fact]
        public void MissingBothNamesBoth()
        {
            var message = UpscaleBackends.MissingComponentsMessage("TensorRT-RTX", false, false);
            Assert.Equal("TensorRT-RTX needs the \"TensorRT-RTX\" and \"TensorRT runtime\" components. Install them in App Settings > Components, or choose another backend.", message);
        }

        // The names the messages use are the component manager's titles.
        [Fact]
        public void MessageNamesMatchComponentTitles()
        {
            Assert.Equal("TensorRT-RTX", new ComponentItem("trt-rtx", 0, false, false).Title);
            Assert.Equal("TensorRT runtime", new ComponentItem("trt-runtime", 0, false, false).Title);
        }

        // trt-rtx is listed on any NVIDIA machine (recommended or not), so it can be installed anywhere.
        [Theory]
        [InlineData("trt-rtx", false, false, true, true)]
        [InlineData("trt-rtx", false, false, false, false)]
        [InlineData("trt-rtx", true, false, false, true)]
        [InlineData("trt-sm89", false, false, true, false)]
        [InlineData("trt-sm120", false, true, true, true)]
        [InlineData("trt-runtime", true, false, false, true)]
        public void ComponentsShown(string name, bool installed, bool recommended, bool nvidia, bool shown) =>
            Assert.Equal(shown, ComponentItem.IsShown(name, installed, recommended, nvidia));

        // The DAT2/HAT variants: bf16 where installed or recommended (an NVIDIA GPU that runs bf16);
        // fp32 where installed, recommended, or while upscales run on DirectML (even on a GPU that
        // runs bf16, where it isn't recommended).
        [Theory]
        [InlineData("models-illustrationjanai-bf16", false, false, true, false, false)]
        [InlineData("models-illustrationjanai-bf16", false, true, true, false, true)]
        [InlineData("models-illustrationjanai-bf16", true, false, true, false, true)]
        [InlineData("models-illustrationjanai-bf16", false, false, true, true, false)]
        [InlineData("models-illustrationjanai-fp32", false, false, true, false, false)]
        [InlineData("models-illustrationjanai-fp32", false, false, true, true, true)]
        [InlineData("models-illustrationjanai-fp32", false, false, false, true, true)]
        [InlineData("models-illustrationjanai-fp32", false, true, false, false, true)]
        [InlineData("models-illustrationjanai-fp32", true, false, true, false, true)]
        [InlineData("models-illustrationjanai", false, true, false, true, true)]
        [InlineData("trt-sm89", false, false, true, true, false)]
        public void PrecisionPacksShown(string name, bool installed, bool recommended, bool nvidia, bool directMl, bool shown) =>
            Assert.Equal(shown, ComponentItem.IsShown(name, installed, recommended, nvidia, directMl));

        [Theory]
        [InlineData("models-mangajanai", "MangaJaNai models (grayscale manga)")]
        [InlineData("models-illustrationjanai", "IllustrationJaNai models (color)")]
        [InlineData("models-illustrationjanai-bf16", "IllustrationJaNai DAT2/HAT, bf16 (NVIDIA)")]
        [InlineData("models-illustrationjanai-fp32", "IllustrationJaNai DAT2/HAT, fp32 (DirectML, older NVIDIA)")]
        [InlineData("trt-sm89", "TensorRT kernels: GeForce RTX 40 series (Ada)")]
        // a pack this version doesn't know shows its id
        [InlineData("models-from-a-newer-version", "models-from-a-newer-version")]
        public void ComponentTitles(string name, string title)
        {
            Assert.Equal(title, new ComponentItem(name, 0, false, false).Title);
            Assert.Equal(title, ComponentItem.TitleOf(name));
        }

        [Theory]
        [InlineData("models-illustrationjanai")]
        [InlineData("models-illustrationjanai-bf16")]
        [InlineData("models-illustrationjanai-fp32")]
        public void ModelPacksHaveDescriptions(string name) =>
            Assert.NotEmpty(new ComponentItem(name, 0, false, false).Description);

        static readonly BackendDevice[] Nvidia =
            [new() { Id = "112600", Name = "NVIDIA GeForce RTX 4090", ComputeCapability = "8.9" },
             new() { Id = "164787", Name = "NVIDIA GeForce RTX 3060", ComputeCapability = "8.6" }];

        // App Settings' setup check: only with DirectML chosen while an NVIDIA GPU is usable.
        [Theory]
        [InlineData("DirectML", "tensorrt-rtx", "164787", "TensorRT-RTX is much faster on NVIDIA GeForce RTX 3060.")]
        [InlineData("DirectML", "tensorrt-rtx", "999", "TensorRT-RTX is much faster on NVIDIA GeForce RTX 4090.")]
        [InlineData("DirectML", "tensorrt", null, "TensorRT is much faster on NVIDIA GeForce RTX 4090.")]
        [InlineData("Auto", "tensorrt-rtx", "112600", null)]
        [InlineData("TensorRT", "tensorrt", "112600", null)]
        [InlineData("TensorRT-RTX", "tensorrt-rtx", "112600", null)]
        [InlineData("DirectML", "directml", "112600", null)]
        [InlineData("DirectML", null, "112600", null)]
        public void FasterBackendHint(string choice, string? nvidiaBackend, string? selected, string? hint) =>
            Assert.Equal(hint, UpscaleBackends.FasterBackendHint(choice, Nvidia, nvidiaBackend, selected));

        [Fact]
        public void NoHintWithoutAnNvidiaGpu()
        {
            Assert.Null(UpscaleBackends.FasterBackendHint("DirectML", [], "tensorrt", null));
            Assert.Null(UpscaleBackends.FasterBackendHint("DirectML", null, "tensorrt", null));
        }

        [Theory]
        [InlineData("Auto", true, false, false, false)]
        [InlineData("TensorRT", false, true, false, false)]
        [InlineData("TensorRT-RTX", false, false, true, false)]
        [InlineData("DirectML", false, false, false, true)]
        // An older version reading a value it doesn't know shows Auto.
        [InlineData("SomethingNew", true, false, false, false)]
        public void ToggleButtons(string saved, bool auto, bool trt, bool rtx, bool dml)
        {
            var vm = new MainWindowViewModel { UpscaleBackend = saved };
            Assert.Equal(new[] { auto, trt, rtx, dml },
                new[] { vm.AutoBackendSelected, vm.TensorRtBackendSelected, vm.TensorRtRtxBackendSelected, vm.DirectMlBackendSelected });
        }

        // The choice is saved as a plain string: "TensorRT-RTX" loads as itself, and the values saved
        // before it existed load unchanged.
        [Theory]
        [InlineData("TensorRT-RTX")]
        [InlineData("TensorRT")]
        [InlineData("DirectML")]
        [InlineData("Auto")]
        public void TheSavedChoiceLoads(string choice)
        {
            var saved = JObject.Parse(File.ReadAllText(NewtonsoftJsonSuspensionDriver.DefaultStatePath));
            saved["UpscaleBackend"] = choice;

            var state = Assert.IsType<MainWindowViewModel>(
                JsonConvert.DeserializeObject<object>(saved.ToString(), NewtonsoftJsonSuspensionDriver.Settings));

            Assert.Equal(choice, state.UpscaleBackend);
        }
    }
}

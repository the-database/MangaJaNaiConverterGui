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
        [InlineData("Auto", new[] { "tensorrt", "directml" })]
        [InlineData("TensorRT", new[] { "tensorrt" })]
        [InlineData("TensorRT-RTX", new[] { "tensorrt-rtx", "tensorrt" })]
        [InlineData("DirectML", new[] { "directml" })]
        // A value from a newer version (or garbage) is Auto, which never asks for TensorRT-RTX.
        [InlineData("SomethingNew", new[] { "tensorrt", "directml" })]
        [InlineData(null, new[] { "tensorrt", "directml" })]
        public void DeviceListQueries(string? choice, string[] backends) =>
            Assert.Equal(backends, UpscaleBackends.DeviceQueryOrder(choice));

        [Theory]
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
            Assert.Equal("TensorRT-RTX needs the \"TensorRT-RTX (experimental)\" component. Install it in App Settings > Components, or choose another backend.", message);
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
            Assert.Equal("TensorRT-RTX needs the \"TensorRT-RTX (experimental)\" and \"TensorRT runtime\" components. Install them in App Settings > Components, or choose another backend.", message);
        }

        // The names the messages use are the component manager's titles.
        [Fact]
        public void MessageNamesMatchComponentTitles()
        {
            Assert.Equal("TensorRT-RTX (experimental)", new ComponentItem("trt-rtx", 0, false, false).Title);
            Assert.Equal("TensorRT runtime", new ComponentItem("trt-runtime", 0, false, false).Title);
        }

        // trt-rtx is never recommended, but an NVIDIA machine lists it so it can be opted into.
        [Theory]
        [InlineData("trt-rtx", false, false, true, true)]
        [InlineData("trt-rtx", false, false, false, false)]
        [InlineData("trt-rtx", true, false, false, true)]
        [InlineData("trt-sm89", false, false, true, false)]
        [InlineData("trt-sm120", false, true, true, true)]
        [InlineData("trt-runtime", true, false, false, true)]
        public void ComponentsShown(string name, bool installed, bool recommended, bool nvidia, bool shown) =>
            Assert.Equal(shown, ComponentItem.IsShown(name, installed, recommended, nvidia));

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

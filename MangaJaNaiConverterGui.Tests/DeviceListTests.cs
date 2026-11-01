using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using System.Linq;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    public class DeviceListTests
    {
        static BackendDevice Device(string id, string name) => new() { Id = id, Name = name };

        [Fact]
        public void ADeviceWithAUniqueNameShowsJustItsName() =>
            Assert.Equal(["NVIDIA GeForce RTX 5090", "Intel Arc A770"],
                BackendDevice.Labels([Device("112600", "NVIDIA GeForce RTX 5090"), Device("98765", "Intel Arc A770")]));

        // Every listed device stays in the list; same-named ones get enough of their id to differ.
        [Fact]
        public void DevicesWithTheSameNameShowTheEndOfTheirIds() =>
            Assert.Equal(["RTX 5090 (#2600)", "RTX 5090 (#4787)", "RTX 5090 (#1705)", "Intel Arc A770"],
                BackendDevice.Labels([Device("112600", "RTX 5090"), Device("164787", "RTX 5090"), Device("161705", "RTX 5090"), Device("98765", "Intel Arc A770")]));

        static readonly BackendDevice[] Twins =
            [Device("9001", "Intel Arc A770"), Device("5555", "RTX 5090"), Device("6666", "RTX 5090")];

        [Fact]
        public void TheSavedIdWins() =>
            Assert.Equal(2, BackendDevice.Find(Twins, "6666", "RTX 5090", 1));

        // DirectML ids change on a reboot: the saved position tells identical cards apart.
        [Fact]
        public void AGoneIdFallsBackToTheSavedPositionWhenTheNameMatches() =>
            Assert.Equal(2, BackendDevice.Find(Twins, "1234", "RTX 5090", 2));

        [Fact]
        public void AGoneIdWithAMovedDeviceFallsBackToTheFirstWithItsName() =>
            Assert.Equal(1, BackendDevice.Find(Twins, "1234", "RTX 5090", 0));

        [Fact]
        public void NothingMatchingFallsBackToTheFirstDevice() =>
            Assert.Equal(0, BackendDevice.Find(Twins, "1234", "Radeon RX 9070", 7));

        [Fact]
        public void TheSuffixGrowsUntilTheIdsDiffer() =>
            Assert.Equal(["RTX 5090 (#112600)", "RTX 5090 (#212600)"],
                BackendDevice.Labels([Device("112600", "RTX 5090"), Device("212600", "RTX 5090")]));

        // --list-devices from mangajanai-cli 0.1.4: compute_capability per device (null on DirectML).
        [Fact]
        public void TheDeviceListHasComputeCapabilities()
        {
            var devices = BackendDevice.ParseList("""
                [{"id": "112600", "name": "NVIDIA GeForce RTX 4090", "vram": 25757220864, "compute_capability": "8.9"},
                 {"id": "98765", "name": "NVIDIA GeForce RTX 2080", "vram": 8589934592, "compute_capability": "7.5"}]
                """);
            Assert.Equal(["8.9", "7.5"], devices.Select(d => d.ComputeCapability));
            Assert.Equal("112600", devices[0].Id);
            Assert.Equal(25757220864, devices[0].Vram);
        }

        [Fact]
        public void DirectMlDevicesHaveNone()
        {
            var devices = BackendDevice.ParseList("""[{"id": "112600", "name": "AMD Radeon RX 9070", "vram": 1, "compute_capability": null}]""");
            Assert.Null(Assert.Single(devices).ComputeCapability);
        }

        // Older backends don't list it: unknown, so bf16 counts as not runnable.
        [Fact]
        public void AnOlderListHasNone()
        {
            var devices = BackendDevice.ParseList("""[{"id": "112600", "name": "NVIDIA GeForce RTX 4090", "vram": 1}]""");
            var device = Assert.Single(devices);
            Assert.Null(device.ComputeCapability);
            Assert.False(ModelVariants.Bf16Runnable("tensorrt", device.ComputeCapability));
        }

        [Theory]
        [InlineData("")]
        [InlineData("ERROR something")]
        [InlineData("{\"id\": 1}")]
        public void SomethingElseIsNoDevices(string output) =>
            Assert.Empty(BackendDevice.ParseList(output));

        [Theory]
        [InlineData("8.9", 8)]
        [InlineData("12.0", 12)]
        [InlineData("7.5", 7)]
        [InlineData(null, null)]
        [InlineData("", null)]
        [InlineData("sm89", null)]
        public void ComputeCapabilityMajor(string? cc, int? major) =>
            Assert.Equal(major, ModelVariants.ComputeCapabilityMajor(cc));

        // --model-info's output.
        [Fact]
        public void ModelInfoWithAHint()
        {
            var info = ModelInfo.Parse("""{"precision": "fp32", "weights": {"fp32": 1000, "fp16": 0, "bf16": 0}, "hint": "fp32 model on NVIDIA GeForce RTX 4090: a bf16 or fp16 export typically runs about 1.5-2x faster."}""" + "\r\n");
            Assert.NotNull(info);
            Assert.Equal("fp32", info.Precision);
            Assert.Equal("fp32 model on NVIDIA GeForce RTX 4090: a bf16 or fp16 export typically runs about 1.5-2x faster.", info.Hint);
        }

        [Fact]
        public void ModelInfoWithoutAHint()
        {
            var info = ModelInfo.Parse("""{"precision": "fp16", "weights": {"fp32": 0, "fp16": 1000, "bf16": 0}, "hint": null}""");
            Assert.NotNull(info);
            Assert.Equal("fp16", info.Precision);
            Assert.Null(info.Hint);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("unknown option --model-info")]
        [InlineData("[1, 2]")]
        public void ModelInfoFromAFailedQueryIsNull(string? output) =>
            Assert.Null(ModelInfo.Parse(output));
    }
}

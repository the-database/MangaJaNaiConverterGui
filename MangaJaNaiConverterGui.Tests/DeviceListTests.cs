using MangaJaNaiConverterGui.ViewModels;
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
    }
}

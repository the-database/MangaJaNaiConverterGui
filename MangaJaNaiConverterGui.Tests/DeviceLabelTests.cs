using MangaJaNaiConverterGui.ViewModels;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    public class DeviceLabelTests
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

        [Fact]
        public void TheSuffixGrowsUntilTheIdsDiffer() =>
            Assert.Equal(["RTX 5090 (#112600)", "RTX 5090 (#212600)"],
                BackendDevice.Labels([Device("112600", "RTX 5090"), Device("212600", "RTX 5090")]));
    }
}

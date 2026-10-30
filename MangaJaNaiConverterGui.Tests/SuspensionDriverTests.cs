using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.ViewModels;
using System;
using System.IO;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    public class SuspensionDriverTests : IDisposable
    {
        readonly string _dir = Path.Combine(Path.GetTempPath(), $"mjn-test-{Path.GetRandomFileName()}");
        readonly string _file;

        public SuspensionDriverTests()
        {
            Directory.CreateDirectory(_dir);
            _file = Path.Combine(_dir, "appstate2.json");
            File.WriteAllText(_file, "saved");
        }

        public void Dispose()
        {
            NewtonsoftJsonSuspensionDriver.StartupComplete = false;
            Directory.Delete(_dir, recursive: true);
        }

        // Regression: any crash used to delete the state, so every workflow reset to defaults.
        [Fact]
        public void CrashAfterStartupKeepsTheState()
        {
            NewtonsoftJsonSuspensionDriver.StartupComplete = true;

            new NewtonsoftJsonSuspensionDriver(_file).InvalidateState();

            Assert.Equal("saved", File.ReadAllText(_file));
            Assert.Equal("saved", File.ReadAllText(_file + ".bak"));
        }

        [Fact]
        public void CrashDuringStartupMovesTheStateAside()
        {
            NewtonsoftJsonSuspensionDriver.StartupComplete = false;

            new NewtonsoftJsonSuspensionDriver(_file).InvalidateState();

            Assert.False(File.Exists(_file));
            Assert.Equal("saved", File.ReadAllText(_file + ".bak"));
        }

        [Fact]
        public void UnreadableStateIsBackedUpBeforeLoadFails()
        {
            File.WriteAllText(_file, "{ not json");

            Assert.ThrowsAny<Exception>(() => new NewtonsoftJsonSuspensionDriver(_file).LoadState());

            Assert.Equal("{ not json", File.ReadAllText(_file + ".bak"));
        }

        [Fact]
        public void SaveReplacesTheStateWithoutLeavingATempFile()
        {
            new NewtonsoftJsonSuspensionDriver(_file).SaveState(new { Value = 42 });

            Assert.Contains("42", File.ReadAllText(_file));
            Assert.False(File.Exists(_file + ".tmp"));
        }

        // The bundled default state is the fallback when the saved state can't be loaded; the app
        // can't start from a view model without workflows.
        [Fact]
        public void DefaultStateHasWorkflows()
        {
            var state = Assert.IsType<MainWindowViewModel>(NewtonsoftJsonSuspensionDriver.LoadDefaultState());

            Assert.NotNull(state.Workflows);
            Assert.NotEmpty(state.Workflows);
        }
    }
}

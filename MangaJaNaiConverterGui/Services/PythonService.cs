using Avalonia.Collections;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.Zip;
using SevenZipExtractor;
using Splat;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.Services
{
    public delegate void ProgressChanged(double percentage);

    // https://github.com/chaiNNer-org/chaiNNer/blob/main/src/main/python/integratedPython.ts
    public class PythonService : IPythonService
    {
        private readonly IUpdateManagerService _updateManagerService;

        public static readonly Dictionary<string, PythonDownload> PYTHON_DOWNLOADS = new()
        {
            {
                "win32",
                new PythonDownload
                {
                    Url = "https://github.com/astral-sh/python-build-standalone/releases/download/20251120/cpython-3.13.9+20251120-x86_64-pc-windows-msvc-install_only.tar.gz",
                    Path = "python/python.exe",
                    Version = "3.13.9",
                    Filename = "Python.tar.gz"
                }
            },
        };

        public Version BackendVersion => new Version(1, 5, 0);

        public string BackendUrl => $"https://github.com/the-database/MangaJaNaiConverterGui-backend/releases/download/{BackendVersion}/mangajanaiconvertergui-backend-{BackendVersion}.7z";

        public PythonService(IUpdateManagerService? updateManagerService = null)
        {
            _updateManagerService = updateManagerService ?? Locator.Current.GetService<IUpdateManagerService>()!;
        }

        public string BackendDirectory => (_updateManagerService?.IsInstalled ?? false) ? Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), @"MangaJaNaiConverterGui") : Path.GetFullPath(@".\backend");

        public string LogsDirectory => Path.Combine(BackendDirectory, "logs");
        public string ModelsDirectory => Path.Combine(BackendDirectory, "models");
        public string PythonDirectory => Path.Combine(BackendDirectory, "python");
        public string PythonBackendVersionPath => Path.Combine(PythonDirectory, "Version.txt");
        public string PythonPath => Path.GetFullPath(Path.Join(PythonDirectory, PYTHON_DOWNLOADS["win32"].Path));

        // MANGAJANAI_STATE_DIR redirects the saved app state (workflows, settings), so test and dev
        // launches never touch a real install's state.
        public string AppStateFolder =>
            Environment.GetEnvironmentVariable("MANGAJANAI_STATE_DIR") is { Length: > 0 } stateDir
                ? stateDir
                : ((_updateManagerService?.IsInstalled ?? false) && !(_updateManagerService?.IsPortable ?? false))
                    ? Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData, Environment.SpecialFolderOption.Create), @"MangaJaNaiConverterGui")
                    : Path.GetFullPath(@".");

        public string AppStateFilename => "appstate2.json";
        public string AppStatePath => Path.Join(AppStateFolder, AppStateFilename);

        public bool IsPythonInstalled() => File.Exists(PythonPath);

        public async Task<bool> IsBackendUpdated()
        {
            if (File.Exists(PythonBackendVersionPath))
            {
                var currentVersion = new Version(await File.ReadAllTextAsync(PythonBackendVersionPath));

                return currentVersion.CompareTo(BackendVersion) >= 0;
            }

            return false;
        }

        public bool AreModelsInstalled() => Directory.Exists(ModelsDirectory) && Directory.GetFiles(ModelsDirectory).Length > 0 && Directory.GetFiles(ModelsDirectory).Any(x => x.Contains("2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16"));

        public class PythonDownload
        {
            public string Url { get; set; }
            public string Version { get; set; }
            public string Path { get; set; }
            public string Filename { get; set; }
        }

        public void ExtractZip(string archivePath, string outFolder, ProgressChanged progressChanged)
        {

            using (var fsInput = File.OpenRead(archivePath))
            using (var zf = new ZipFile(fsInput))
            {

                for (var i = 0; i < zf.Count; i++)
                {
                    ZipEntry zipEntry = zf[i];

                    if (!zipEntry.IsFile)
                    {
                        // Ignore directories
                        continue;
                    }
                    String entryFileName = zipEntry.Name;
                    // to remove the folder from the entry:
                    //entryFileName = Path.GetFileName(entryFileName);
                    // Optionally match entrynames against a selection list here
                    // to skip as desired.
                    // The unpacked length is available in the zipEntry.Size property.

                    // Manipulate the output filename here as desired.
                    var fullZipToPath = Path.Combine(outFolder, entryFileName);
                    var directoryName = Path.GetDirectoryName(fullZipToPath);
                    if (directoryName.Length > 0)
                    {
                        Directory.CreateDirectory(directoryName);
                    }

                    // 4K is optimum
                    var buffer = new byte[4096];

                    // Unzip file in buffered chunks. This is just as fast as unpacking
                    // to a buffer the full size of the file, but does not waste memory.
                    // The "using" will close the stream even if an exception occurs.
                    using (var zipStream = zf.GetInputStream(zipEntry))
                    using (Stream fsOutput = File.Create(fullZipToPath))
                    {
                        StreamUtils.Copy(zipStream, fsOutput, buffer);
                    }

                    var percentage = Math.Round((double)i / zf.Count * 100, 0);
                    progressChanged?.Invoke(percentage);
                }
            }
        }

        public void Extract7z(string archiveName, string outFolder)
        {
            using ArchiveFile archiveFile = new(archiveName);
            archiveFile.Extract(outFolder);
        }

        private AvaloniaList<string>? _allModels;

        public AvaloniaList<string> AllModels
        {
            get
            {
                if (_allModels == null)
                {

                    try
                    {
                        var models = new AvaloniaList<string>(Directory.GetFiles(ModelsDirectory).Where(filename =>
                            Path.GetExtension(filename).Equals(".pth", StringComparison.CurrentCultureIgnoreCase) ||
                            Path.GetExtension(filename).Equals(".pt", StringComparison.CurrentCultureIgnoreCase) ||
                            Path.GetExtension(filename).Equals(".ckpt", StringComparison.CurrentCultureIgnoreCase) ||
                            Path.GetExtension(filename).Equals(".safetensors", StringComparison.CurrentCultureIgnoreCase)
                        )
                        .Select(filename => Path.GetFileName(filename))
                        .Order().ToList());

                        models.Add("No Model");

                        Debug.WriteLine($"GetAllModels: {models.Count}");

                        _allModels = models;
                    }
                    catch (DirectoryNotFoundException)
                    {
                        Debug.WriteLine($"GetAllModels: DirectoryNotFoundException");
                        return [];
                    }
                }

                return _allModels;
            }
        }
    }
}

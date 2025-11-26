using Avalonia.Collections;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
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

        public Version BackendVersion => new Version(1, 3, 0);

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

        public string AppStateFolder => ((_updateManagerService?.IsInstalled ?? false) && !(_updateManagerService?.IsPortable ?? false)) ? Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), @"MangaJaNaiConverterGui") : Path.GetFullPath(@".");

        public string AppStateFilename => "appstate2.json";
        public string AppStatePath => Path.Join(AppStateFolder, AppStateFilename);

        public bool IsPythonInstalled() => File.Exists(PythonPath);

        public async Task<bool> IsPythonUpdated()
        {
            var relPythonPath = @".\python\python\python.exe";

            var cmd = $@"{relPythonPath} -V";

            // Create a new process to run the CMD command
            using (var process = new Process())
            {
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.Arguments = @$"/C {cmd}";
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;
                process.StartInfo.WorkingDirectory = BackendDirectory;

                Version? result = null;

                // Create a StreamWriter to write the output to a log file
                try
                {
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            // ignore
                        }
                    };

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            result = new Version(e.Data.Replace("Python ", ""));
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();
                }
                catch (IOException) { }

                if (result == null || result.CompareTo(new Version(PYTHON_DOWNLOADS["win32"].Version)) < 0)
                {
                    return false;
                }
            }

            return true;
        }

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

        public void ExtractTgz(string gzArchiveName, string destFolder)
        {
            Stream inStream = File.OpenRead(gzArchiveName);
            Stream gzipStream = new GZipInputStream(inStream);

            TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream, Encoding.UTF8);
            tarArchive.ExtractContents(destFolder);
            tarArchive.Close();

            gzipStream.Close();
            inStream.Close();
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

        public void AddPythonPth(string destFolder)
        {
            string[] lines = { "python313.zip", "DLLs", "Lib", ".", "Lib/site-packages" };
            var filename = "python313._pth";

            using var outputFile = new StreamWriter(Path.Combine(destFolder, filename));

            foreach (string line in lines)
                outputFile.WriteLine(line);
        }

        public string InstallUpdatePythonDependenciesCommand
        {
            get
            {
                var relPythonPath = @".\python\python\python.exe";

                return $@"{relPythonPath} -m pip install -U pip wheel --no-warn-script-location && {relPythonPath} -m pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location && {relPythonPath} -m pip install ""{Path.GetFullPath(@".\backend\src")}"" --no-warn-script-location";
            }
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

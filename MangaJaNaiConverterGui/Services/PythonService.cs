using Avalonia.Collections;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using ICSharpCode.SharpZipLib.Zip;
using Splat;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

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
                    Url = "https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-x86_64-pc-windows-msvc-shared-install_only.tar.gz",
                    Path = "python/python.exe",
                    Version = "3.11.9",
                    Filename = "Python.tar.gz"
                }
            },
        };

        public PythonService(IUpdateManagerService? updateManagerService = null)
        {
            _updateManagerService = updateManagerService ?? Locator.Current.GetService<IUpdateManagerService>()!;
        }

        public string BackendDirectory => (_updateManagerService?.IsInstalled ?? false) ? Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), @"MangaJaNaiConverterGui") : Path.GetFullPath(@".\backend");

        public string ModelsDirectory => Path.Combine(BackendDirectory, "models");
        public string PythonDirectory => Path.Combine(BackendDirectory, "python");
        public string PythonPath => Path.GetFullPath(Path.Join(PythonDirectory, PYTHON_DOWNLOADS["win32"].Path));

        public bool IsPythonInstalled() => File.Exists(PythonPath);
        public bool AreModelsInstalled() => Directory.Exists(ModelsDirectory) && Directory.GetFiles(ModelsDirectory).Length > 0;

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

        public void AddPythonPth(string destFolder)
        {
            string[] lines = { "python311.zip", "DLLs", "Lib", ".", "Lib/site-packages" };
            var filename = "python311._pth";

            using var outputFile = new StreamWriter(Path.Combine(destFolder, filename));

            foreach (string line in lines)
                outputFile.WriteLine(line);
        }

        public string InstallUpdatePythonDependenciesCommand
        {
            get
            {
                string[] dependencies = {
                    "spandrel==0.3.4",
                    "spandrel_extra_arches==0.1.1",
                    "opencv-python==4.10.0.84",
                    "pillow-avif-plugin==1.4.6",
                    "rarfile==4.2",
                    "multiprocess==0.70.16",
                    "chainner_ext==0.3.10",
                    "sanic==24.6.0",
                    "pynvml==11.5.3",
                    "psutil==6.0.0"
                };

                var relPythonPath = @".\python\python\python.exe";

                return $@"{relPythonPath} -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 && {relPythonPath} -m pip install {string.Join(" ", dependencies)}";
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

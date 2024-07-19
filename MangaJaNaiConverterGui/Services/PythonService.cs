using Splat;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.IO.Compression;
using System.Net;
using OctaneEngineCore;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using Newtonsoft.Json;
using System.Diagnostics;

namespace MangaJaNaiConverterGui.Services
{
    // https://github.com/chaiNNer-org/chaiNNer/blob/main/src/main/python/integratedPython.ts
    public class PythonService : IPythonService
    {
        private readonly IUpdateManagerService _updateManagerService;
        private readonly IEngine _engine;

        private static readonly Dictionary<string, PythonDownload> DOWNLOADS = new ()
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

        public PythonService(IUpdateManagerService? updateManagerService = null, IEngine? engine = null)
        {
            _updateManagerService = updateManagerService ?? Locator.Current.GetService<IUpdateManagerService>();
            _engine = engine ?? Locator.Current.GetService<IEngine>();
        }

        public string PythonDirectory => (_updateManagerService?.IsInstalled ?? false) ? Path.GetFullPath(@"%APPDATA%\MangaJaNaiConverterGui\python") : Path.GetFullPath(@".\backend\python");
        public string PythonPath => Path.Join(PythonDirectory, DOWNLOADS["win32"].Path);

        public bool IsPythonInstalled()
        {
            return File.Exists(PythonPath);
        }

        public async Task InstallPython()
        {
            // Download Python tgz
            var download = DOWNLOADS["win32"];
            var targetPath = Path.Join(PythonDirectory, download.Filename);
            Directory.CreateDirectory(PythonDirectory);
            _engine.DownloadFile(download.Url, targetPath, null, null).Wait();
    
            // Extract Python tgz
            ExtractTGZ(targetPath, PythonDirectory);

            // Delete Python tgz
            File.Delete(targetPath);

            // Add Python _pth file to Python install
            AddPythonPth(Path.GetDirectoryName(PythonPath));

            // Install dependencies
            await InstallUpdatePythonDependencies();
        }

        class PythonDownload
        {
            public string Url { get; set; }
            public string Version { get; set; }
            public string Path { get; set; }
            public string Filename { get; set; }
        }

        private static void ExtractTGZ(string gzArchiveName, string destFolder)
        {
            Stream inStream = File.OpenRead(gzArchiveName);
            Stream gzipStream = new GZipInputStream(inStream);

            TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream, Encoding.UTF8);
            tarArchive.ExtractContents(destFolder);
            tarArchive.Close();

            gzipStream.Close();
            inStream.Close();
        }

        private static void AddPythonPth(string destFolder)
        {
            string[] lines = { "python311.zip", "DLLs", "Lib", ".", "Lib/site-packages" };
            var filename = "python311._pth";

            using var outputFile = new StreamWriter(Path.Combine(destFolder, filename));
            
            foreach (string line in lines)
                outputFile.WriteLine(line);            
        }

        public async Task<string[]> InstallUpdatePythonDependencies()
        {
            string[] dependencies = { 
                "spandrel>=0.3.4",
                //"spandrel_extra_arches>=0.1.1",
                //"opencv-python>=4.10.0.84",
                //"pillow-avif-plugin>=1.4.6",
                //"rarfile>=4.2",
                //"multiprocess>=0.70.16",
                //"chainner_ext>=0.3.10",
                //"sanic>=24.6.0",
                //"pynvml>=11.5.3",
                //"psutil>=6.0.0" 
            };

            //var cmd = $@"{PythonPath} -m pip install torch>=2.3.1 torchvision>=0.18.1 --index-url https://download.pytorch.org/whl/cu121 && {PythonPath} -m pip install {string.Join(" ", dependencies)}";
            var cmd = $@"{PythonPath} -m pip install {string.Join(" ", dependencies)}";
            Debug.WriteLine(cmd);

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

                var result = string.Empty;

                // Create a StreamWriter to write the output to a log file
                try
                {
                    //using var outputFile = new StreamWriter("error.log", append: true);
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            Debug.WriteLine($"STDERR = {e.Data}");
                        }
                    };

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            result = e.Data;
                            Debug.WriteLine($"STDOUT = {e.Data}");
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();

                    if (!string.IsNullOrEmpty(result))
                    {
                        Debug.WriteLine($"Result = {result}");
                        //return JsonConvert.DeserializeObject<string[]>(result);
                    }
                }
                catch (IOException) { }
            }

            return [];
        }
    }
}

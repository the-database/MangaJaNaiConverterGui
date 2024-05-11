using Avalonia.Collections;
using Avalonia.Data;
using Avalonia.Threading;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Progression.Extras;
using ReactiveUI;
using SevenZipExtractor;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Velopack;
using Velopack.Sources;

namespace MangaJaNaiConverterGui.ViewModels
{
    [DataContract]
    public class MainWindowViewModel : ViewModelBase
    {
        public static readonly List<string> IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
        public static readonly List<string> ARCHIVE_EXTENSIONS = [".zip", ".cbz", ".rar", ".cbr"];

        private readonly DispatcherTimer _timer = new();

        private readonly UpdateManager _um;
        private UpdateInfo? _update = null;

        public MainWindowViewModel()
        {
            _timer.Interval = TimeSpan.FromSeconds(1);
            _timer.Tick += _timer_Tick;

            ShowDialog = new Interaction<MainWindowViewModel, MainWindowViewModel?>();

            _um = new UpdateManager(new GithubSource("https://github.com/the-database/MangaJaNaiConverterGui", null, false));

            CheckForUpdates();
        }

        private string[] _commonResolutions = [
"0x0",
"0x1250",
"0x1251",
"0x1350",
"0x1351",
"0x1450",
"0x1451",
"0x1550",
"0x1551",
"0x1760",
"0x1761",
"0x1984",
"0x1985",];

        public string[] CommonResolutions
        {
            get => _commonResolutions;
            set => this.RaiseAndSetIfChanged(ref _commonResolutions, value);
        }

        public Interaction<MainWindowViewModel, MainWindowViewModel?> ShowDialog { get; }

        private void _timer_Tick(object? sender, EventArgs e)
        {
            ElapsedTime = ElapsedTime.Add(TimeSpan.FromSeconds(1));
        }

        private CancellationTokenSource? _cancellationTokenSource;
        private Process? _runningProcess = null;
        private readonly IETACalculator _archiveEtaCalculator = new ETACalculator(2, 3.0);
        private readonly IETACalculator _totalEtaCalculator = new ETACalculator(2, 3.0);

        public TimeSpan ArchiveEtr => _archiveEtaCalculator.ETAIsAvailable ? _archiveEtaCalculator.ETR : TimeSpan.FromSeconds(0);
        public string ArchiveEta => _archiveEtaCalculator.ETAIsAvailable ? _archiveEtaCalculator.ETA.ToString("t") : "please wait";

        public TimeSpan TotalEtr => _totalEtaCalculator.ETAIsAvailable ? _totalEtaCalculator.ETR : ArchiveEtr + (ElapsedTime + ArchiveEtr) * (ProgressTotalFiles - (ProgressCurrentFile + 1));

        public string TotalEta => _totalEtaCalculator.ETAIsAvailable ? _totalEtaCalculator.ETA.ToString("t") : _archiveEtaCalculator.ETAIsAvailable ? DateTime.Now.Add(TotalEtr).ToString("t") : "please wait";

        public bool IsInstalled => _um?.IsInstalled ?? false;

        private bool _showCheckUpdateButton = true;
        public bool ShowCheckUpdateButton
        {
            get => _showCheckUpdateButton;
            set => this.RaiseAndSetIfChanged(ref _showCheckUpdateButton, value);
        }

        private bool _showDownloadButton = false;
        public bool ShowDownloadButton
        {
            get => _showDownloadButton;
            set
            {
                this.RaiseAndSetIfChanged(ref _showDownloadButton, value);
                this.RaisePropertyChanged(nameof(ShowCheckUpdateButton));
            }
        }

        private bool _showApplyButton = false;
        public bool ShowApplyButton
        {
            get => _showApplyButton;
            set
            {
                this.RaiseAndSetIfChanged(ref _showApplyButton, value);
                this.RaisePropertyChanged(nameof(ShowCheckUpdateButton));
            }
        }

        public string AppVersion => _um?.CurrentVersion?.ToString() ?? "";

        private string _updateStatusText = string.Empty;
        public string UpdateStatusText
        {
            get => _updateStatusText;
            set => this.RaiseAndSetIfChanged(ref _updateStatusText, value);
        }

        private string[] _tileSizes = [
            "Auto (Estimate)",
            "Maximum",
            "No Tiling",
            "128",
            "192",
            "256",
            "384",
            "512",
            "768",
            "1024",
            "2048",
            "4096"];

        public string[] TileSizes
        {
            get => _tileSizes;
            set => this.RaiseAndSetIfChanged(ref _tileSizes, value);
        }

        private string[] _deviceList = [];

        public string[] DeviceList
        {
            get => _deviceList;
            set
            {
                this.RaiseAndSetIfChanged(ref _deviceList, value);
                this.RaisePropertyChanged(nameof(SelectedDeviceIndex));
            }
        }

        private bool _autoUpdate;
        [DataMember]
        public bool AutoUpdateEnabled
        {
            get => _autoUpdate;
            set => this.RaiseAndSetIfChanged(ref _autoUpdate, value);
        }

        private int _selectedDeviceIndex;
        [DataMember]
        public int SelectedDeviceIndex
        {
            get => _selectedDeviceIndex;
            set => this.RaiseAndSetIfChanged(ref _selectedDeviceIndex, value);
        }

        private bool _useCpu;
        [DataMember]
        public bool UseCpu
        {
            get => _useCpu;
            set => this.RaiseAndSetIfChanged(ref _useCpu, value);
        }

        private bool _useFp16;
        [DataMember]
        public bool UseFp16
        {
            get => _useFp16;
            set => this.RaiseAndSetIfChanged(ref _useFp16, value);
        }


        private bool _upscaling = false;
        [IgnoreDataMember]
        public bool Upscaling
        {
            get => _upscaling;
            set
            {
                this.RaiseAndSetIfChanged(ref _upscaling, value);
                this.RaisePropertyChanged(nameof(UpscaleEnabled));
                this.RaisePropertyChanged(nameof(LeftStatus));
            }
        }

        private string _validationText = string.Empty;
        public string ValidationText
        {
            get => _validationText;
            set
            {
                this.RaiseAndSetIfChanged(ref _validationText, value);
                this.RaisePropertyChanged(nameof(LeftStatus));
            }
        }

        public string ConsoleText => string.Join("\n", ConsoleQueue);

        private static readonly int CONSOLE_QUEUE_CAPACITY = 1000;

        private ConcurrentQueue<string> _consoleQueue = new();
        public ConcurrentQueue<string> ConsoleQueue
        {
            get => this._consoleQueue;
            set
            {
                this.RaiseAndSetIfChanged(ref _consoleQueue, value);
                this.RaisePropertyChanged(nameof(ConsoleText));
            }
        }

        private bool _showConsole = false;
        public bool ShowConsole
        {
            get => _showConsole;
            set => this.RaiseAndSetIfChanged(ref _showConsole, value);
        }

        private bool _showAppSettings = false;
        public bool RequestShowAppSettings
        {
            get => _showAppSettings;
            set
            {
                this.RaiseAndSetIfChanged(ref _showAppSettings, value);
                this.RaisePropertyChanged(nameof(ShowAppSettings));
                this.RaisePropertyChanged(nameof(ShowMainForm));
            }
        }

        private bool _isExtractingBackend = false;
        public bool IsExtractingBackend
        {
            get => _isExtractingBackend;
            set
            {
                this.RaiseAndSetIfChanged(ref _isExtractingBackend, value);
                this.RaisePropertyChanged(nameof(RequestShowAppSettings));
                this.RaisePropertyChanged(nameof(ShowMainForm));
            }
        }

        public bool ShowAppSettings => RequestShowAppSettings && !IsExtractingBackend;

        public bool ShowMainForm => !RequestShowAppSettings && !IsExtractingBackend;

        private bool _showEstimates = false;
        public bool ShowEstimates
        {
            get => _showEstimates;
            set => this.RaiseAndSetIfChanged(ref _showEstimates, value);
        }

        private string _inputStatusText = string.Empty;
        public string InputStatusText
        {
            get => _inputStatusText;
            set
            {
                this.RaiseAndSetIfChanged(ref _inputStatusText, value);
                this.RaisePropertyChanged(nameof(LeftStatus));
            }
        }

        public string LeftStatus => !CurrentWorkflow.Valid ? ValidationText.Replace("\n", " ") : $"{InputStatusText} selected for upscaling.";

        private int _progressCurrentFile = 0;
        public int ProgressCurrentFile
        {
            get => _progressCurrentFile;
            set => this.RaiseAndSetIfChanged(ref _progressCurrentFile, value);
        }

        private int _progressTotalFiles = 0;
        public int ProgressTotalFiles
        {
            get => _progressTotalFiles;
            set => this.RaiseAndSetIfChanged(ref _progressTotalFiles, value);
        }

        private int _progressCurrentFileInCurrentArchive = 0;
        public int ProgressCurrentFileInArchive
        {
            get => _progressCurrentFileInCurrentArchive;
            set => this.RaiseAndSetIfChanged(ref _progressCurrentFileInCurrentArchive, value);
        }

        private int _progressTotalFilesInCurrentArchive = 0;
        public int ProgressTotalFilesInCurrentArchive
        {
            get => _progressTotalFilesInCurrentArchive;
            set => this.RaiseAndSetIfChanged(ref _progressTotalFilesInCurrentArchive, value);
        }

        private bool _showArchiveProgressBar = false;
        public bool ShowArchiveProgressBar
        {
            get => _showArchiveProgressBar;
            set => this.RaiseAndSetIfChanged(ref _showArchiveProgressBar, value);
        }

        public bool UpscaleEnabled => CurrentWorkflow.Valid && !Upscaling;

        private TimeSpan _elapsedTime = TimeSpan.FromSeconds(0);
        public TimeSpan ElapsedTime
        {
            get => _elapsedTime;
            set
            {
                this.RaiseAndSetIfChanged(ref _elapsedTime, value);
            }
        }

        private AvaloniaList<UpscaleWorkflow> _workflows;
        [DataMember]
        public AvaloniaList<UpscaleWorkflow> Workflows
        {
            get => _workflows;
            set => this.RaiseAndSetIfChanged(ref _workflows, value);
        }

        public AvaloniaList<UpscaleWorkflow> CustomWorkflows => new AvaloniaList<UpscaleWorkflow>(Workflows.Skip(1).ToList());

        private int _selectedWorkflowIndex = 0;
        [DataMember]
        public int SelectedWorkflowIndex
        {
            get => _selectedWorkflowIndex;
            set 
            { 
                this.RaiseAndSetIfChanged(ref _selectedWorkflowIndex, value); 
                this.RaisePropertyChanged(nameof(CurrentWorkflow));
                this.RaisePropertyChanged(nameof(CurrentWorkflow.ActiveWorkflow));
            }
        }

        public UpscaleWorkflow CurrentWorkflow
        {
            get => Workflows[SelectedWorkflowIndex];
        }

        public void HandleWorkflowSelected(int workflowIndex)
        {
            SelectedWorkflowIndex = workflowIndex;
            RequestShowAppSettings = false;
        }

        public void HandleAppSettingsSelected()
        {
            RequestShowAppSettings = true;
        }


        public async Task RunUpscale()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;

            var task = Task.Run(async () =>
            {
                ElapsedTime = TimeSpan.FromSeconds(0);
                ShowEstimates = true;
                _archiveEtaCalculator.Reset();
                _totalEtaCalculator.Reset();
                ct.ThrowIfCancellationRequested();
                ConsoleQueueClear();
                Upscaling = true;
                ProgressCurrentFile = 0;
                ProgressCurrentFileInArchive = 0;
                ShowArchiveProgressBar = false;

                var flags = new StringBuilder();
                if (CurrentWorkflow.UpscaleArchives)
                {
                    flags.Append("--upscale-archives ");
                }
                if (CurrentWorkflow.UpscaleImages)
                {
                    flags.Append("--upscale-images ");
                }
                if (CurrentWorkflow.OverwriteExistingFiles)
                {
                    flags.Append("--overwrite-existing-files ");
                }
                //if (AutoAdjustLevels)
                //{
                //    flags.Append("--auto-adjust-levels ");
                //}
                if (CurrentWorkflow.UseLosslessCompression)
                {
                    flags.Append("--use-lossless-compression ");
                }
                if (UseCpu)
                {
                    flags.Append("--use-cpu ");
                }
                if (UseFp16)
                {
                    flags.Append("--use-fp16 ");
                }

                var inputArgs = $"--input-file-path \"{CurrentWorkflow.InputFilePath}\" ";

                if (CurrentWorkflow.SelectedTabIndex == 1)
                {
                    inputArgs = $"--input-folder-path \"{CurrentWorkflow.InputFolderPath}\" ";
                }

                var grayscaleModelFilePath = "";// TODO string.IsNullOrWhiteSpace(GrayscaleModelFilePath) ? GrayscaleModelFilePath : Path.GetFullPath(GrayscaleModelFilePath);
                var colorModelFilePath = "";// TODO string.IsNullOrWhiteSpace(ColorModelFilePath) ? ColorModelFilePath : Path.GetFullPath(ColorModelFilePath);

                var cmd = ""; // TODO  $@".\python\python.exe "".\backend\src\runmangajanaiconverterguiupscale.py"" --selected-device {SelectedDeviceIndex} {inputArgs} --output-folder-path ""{OutputFolderPath}"" --output-filename ""{OutputFilename}"" --resize-height-before-upscale {ResizeHeightBeforeUpscale} --resize-width-before-upscale {ResizeWidthBeforeUpscale} --resize-factor-before-upscale {ResizeFactorBeforeUpscale} --grayscale-model-path ""{grayscaleModelFilePath}"" --grayscale-model-tile-size ""{GrayscaleModelTileSize}"" --color-model-path ""{colorModelFilePath}"" --color-model-tile-size ""{ColorModelTileSize}"" --image-format {ImageFormat} --lossy-compression-quality {LossyCompressionQuality} --resize-height-after-upscale {ResizeHeightAfterUpscale} --resize-width-after-upscale {ResizeWidthAfterUpscale} --resize-factor-after-upscale {ResizeFactorAfterUpscale} {flags}";
                ConsoleQueueEnqueue($"Upscaling with command: {cmd}");
                await RunCommand($@" /C {cmd}");

                CurrentWorkflow.Valid = true;
            }, ct);

            try
            {
                _timer.Start();
                await task;
                _timer.Stop();
                CurrentWorkflow.Validate();
            }
            catch (OperationCanceledException e)
            {
                _timer.Stop();
                Console.WriteLine($"{nameof(OperationCanceledException)} thrown with message: {e.Message}");
                Upscaling = false;
            }
            finally
            {
                _timer.Stop();
                _cancellationTokenSource.Dispose();
                Upscaling = false;
            }
        }

        public void CancelUpscale()
        {
            try
            {
                _cancellationTokenSource?.Cancel();
                if (_runningProcess != null && !_runningProcess.HasExited)
                {
                    // Kill the process
                    _runningProcess.Kill(true);
                    _runningProcess = null; // Clear the reference to the terminated process
                }
                CurrentWorkflow.Validate();
            }
            catch { }
        }


        private void CheckInputs()
        {
            if (CurrentWorkflow.Valid && !Upscaling)
            {
                var overwriteText = CurrentWorkflow.OverwriteExistingFiles ? "overwritten" : "skipped";

                // input file
                if (CurrentWorkflow.SelectedTabIndex == 0)
                {
                    StringBuilder status = new();
                    var skipFiles = 0;



                    if (IMAGE_EXTENSIONS.Any(x => CurrentWorkflow.InputFilePath.ToLower().EndsWith(x)))
                    {
                        var outputFilePath = Path.ChangeExtension(
                                                Path.Join(
                                                    Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                    CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(CurrentWorkflow.InputFilePath))),
                                                CurrentWorkflow.ImageFormat);
                        if (File.Exists(outputFilePath))
                        {
                            status.Append($" (1 image already exists and will be {overwriteText})");
                            if (!CurrentWorkflow.OverwriteExistingFiles)
                            {
                                skipFiles++;
                            }
                        }
                    }
                    else if (ARCHIVE_EXTENSIONS.Any(x => CurrentWorkflow.InputFilePath.ToLower().EndsWith(x)))
                    {
                        var outputFilePath = Path.ChangeExtension(
                                                Path.Join(
                                                    Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                    CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(CurrentWorkflow.InputFilePath))),
                                                "cbz");
                        if (File.Exists(outputFilePath))
                        {
                            status.Append($" (1 archive already exists and will be {overwriteText})");
                            if (!CurrentWorkflow.OverwriteExistingFiles)
                            {
                                skipFiles++;
                            }
                        }
                    }
                    else
                    {
                        // TODO ???
                    }

                    var s = skipFiles > 0 ? "s" : "";
                    if (IMAGE_EXTENSIONS.Any(x => CurrentWorkflow.InputFilePath.ToLower().EndsWith(x)))
                    {
                        status.Insert(0, $"{1 - skipFiles} image{s}");
                    }
                    else if (ARCHIVE_EXTENSIONS.Any(x => CurrentWorkflow.InputFilePath.ToLower().EndsWith(x)))
                    {
                        status.Insert(0, $"{1 - skipFiles} archive{s}");
                    }
                    else
                    {
                        status.Insert(0, "0 files");
                    }

                    InputStatusText = status.ToString();
                    ProgressCurrentFile = 0;
                    ProgressTotalFiles = 1 - skipFiles;
                    ProgressCurrentFileInArchive = 0;
                    ProgressTotalFilesInCurrentArchive = 0;
                    ShowArchiveProgressBar = false;
                }
                else  // input folder
                {
                    List<string> statuses = new();
                    var existImageCount = 0;
                    var existArchiveCount = 0;
                    var totalFileCount = 0;

                    if (CurrentWorkflow.UpscaleImages)
                    {
                        var images = Directory.EnumerateFiles(CurrentWorkflow.InputFolderPath, "*.*", SearchOption.AllDirectories)
                            .Where(file => IMAGE_EXTENSIONS.Any(ext => file.ToLower().EndsWith(ext)));
                        var imagesCount = 0;

                        foreach (var inputImagePath in images)
                        {
                            var outputImagePath = Path.ChangeExtension(
                                                    Path.Join(
                                                        Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                        CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputImagePath))),
                                                    CurrentWorkflow.ImageFormat);
                            // if out file exists, exist count ++
                            // if overwrite image OR out file doesn't exist, count image++
                            var fileExists = File.Exists(outputImagePath);

                            if (fileExists)
                            {
                                existImageCount++;
                            }

                            if (!fileExists || CurrentWorkflow.OverwriteExistingFiles)
                            {
                                imagesCount++;
                            }
                        }

                        var imageS = imagesCount == 1 ? "" : "s";
                        var existImageS = existImageCount == 1 ? "" : "s";

                        statuses.Add($"{imagesCount} image{imageS} ({existImageCount} image{existImageS} already exist and will be {overwriteText})");
                        totalFileCount += imagesCount;
                    }
                    if (CurrentWorkflow.UpscaleArchives)
                    {
                        var archives = Directory.EnumerateFiles(CurrentWorkflow.InputFolderPath, "*.*", SearchOption.AllDirectories)
                            .Where(file => ARCHIVE_EXTENSIONS.Any(ext => file.ToLower().EndsWith(ext)));
                        var archivesCount = 0;

                        foreach (var inputArchivePath in archives)
                        {
                            var outputArchivePath = Path.ChangeExtension(
                                                        Path.Join(
                                                            Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                            CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputArchivePath))),
                                                        "cbz");
                            var fileExists = File.Exists(outputArchivePath);

                            if (fileExists)
                            {
                                existArchiveCount++;
                            }

                            if (!fileExists || CurrentWorkflow.OverwriteExistingFiles)
                            {
                                archivesCount++;
                            }
                        }

                        var archiveS = archivesCount == 1 ? "" : "s";
                        var existArchiveS = existArchiveCount == 1 ? "" : "s";
                        statuses.Add($"{archivesCount} archive{archiveS} ({existArchiveCount} archive{existArchiveS} already exist and will be {overwriteText})");
                        totalFileCount += archivesCount;
                    }

                    if (!CurrentWorkflow.UpscaleArchives && !CurrentWorkflow.UpscaleImages)
                    {
                        InputStatusText = "0 files";
                    }
                    else
                    {
                        InputStatusText = $"{string.Join(" and ", statuses)}";
                    }

                    ProgressCurrentFile = 0;
                    ProgressTotalFiles = totalFileCount;
                    ProgressCurrentFileInArchive = 0;
                    ProgressTotalFilesInCurrentArchive = 0;
                    ShowArchiveProgressBar = false;

                }
            }
        }

        public void AddChain()
        {
            CurrentWorkflow.Chains.Add(new UpscaleChain());
            UpdateChainHeaders();
        }

        public void DeleteChain(UpscaleChain chain)
        {
            try
            {
                CurrentWorkflow.Chains.Remove(chain);
            }
            catch (ArgumentOutOfRangeException)
            {

            }

            UpdateChainHeaders();
        }

        public void UpdateChainHeaders()
        {
            for (var i = 0; i < CurrentWorkflow.Chains.Count; i++)
            {
                CurrentWorkflow.Chains[i].ChainNumber = (i + 1).ToString();
            }
        }



        public async Task RunCommand(string command)
        {
            // Create a new process to run the CMD command
            using (var process = new Process())
            {
                _runningProcess = process;
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.Arguments = command;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.WorkingDirectory = Path.GetFullPath(@".\chaiNNer");
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;

                // Create a StreamWriter to write the output to a log file
                using (var outputFile = new StreamWriter("error.log", append: true))
                {
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            outputFile.WriteLine(e.Data); // Write the output to the log file
                            ConsoleQueueEnqueue(e.Data);
                        }
                    };

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            if (e.Data.StartsWith("PROGRESS="))
                            {
                                if (e.Data.Contains("_zip_image"))
                                {
                                    ShowArchiveProgressBar = true;
                                    ProgressCurrentFileInArchive++;
                                    UpdateEtas();
                                }
                                else
                                {
                                    ProgressCurrentFile++;
                                    UpdateEtas();

                                }
                            }
                            else if (e.Data.StartsWith("TOTALZIP="))
                            {
                                if (int.TryParse(e.Data.Replace("TOTALZIP=", ""), out var total))
                                {
                                    ShowArchiveProgressBar = true;
                                    ProgressCurrentFileInArchive = 0;
                                    ProgressTotalFilesInCurrentArchive = total;
                                    UpdateEtas();
                                }
                            }
                            else
                            {
                                outputFile.WriteLine(e.Data); // Write the output to the log file
                                ConsoleQueueEnqueue(e.Data);
                                Debug.WriteLine(e.Data);
                            }
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();
                }

            }
        }

        public async Task<string[]> InitializeDeviceList()
        {
            if (!File.Exists(@".\chaiNNer\backend\src\device_list.py"))
            {
                return [];
            }

            // Create a new process to run the CMD command
            using (var process = new Process())
            {
                _runningProcess = process;
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.Arguments = @"/C .\python\python.exe .\backend\src\device_list.py";
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.WorkingDirectory = Path.GetFullPath(@".\chaiNNer");
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;

                var result = string.Empty;

                // Create a StreamWriter to write the output to a log file
                try
                {
                    using var outputFile = new StreamWriter("error.log", append: true);
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            //outputFile.WriteLine(e.Data); // Write the output to the log file
                            //ConsoleQueueEnqueue(e.Data);
                            Debug.WriteLine(e.Data);
                        }
                    };

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            result = e.Data;
                            Debug.WriteLine(e.Data);
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();

                    if (!string.IsNullOrEmpty(result))
                    {
                        return JsonConvert.DeserializeObject<string[]>(result);
                    }
                }
                catch (IOException) { }
            }

            return [];
        }

        public async void ShowSettingsDialog()
        {
            var result = await ShowDialog.Handle(this);
        }

        private void UpdateEtas()
        {
            if (ProgressTotalFilesInCurrentArchive > 0)
            {
                _archiveEtaCalculator.Update(ProgressCurrentFileInArchive / (float)ProgressTotalFilesInCurrentArchive);
            }

            if (ProgressTotalFiles > 0)
            {
                _totalEtaCalculator.Update(ProgressCurrentFile / (float)ProgressTotalFiles);
            }

            this.RaisePropertyChanged(nameof(ArchiveEtr));
            this.RaisePropertyChanged(nameof(ArchiveEta));
            this.RaisePropertyChanged(nameof(TotalEtr));
            this.RaisePropertyChanged(nameof(TotalEta));
        }

        private void ConsoleQueueClear()
        {
            ConsoleQueue.Clear();
            this.RaisePropertyChanged(nameof(ConsoleText));
        }

        private void ConsoleQueueEnqueue(string value)
        {
            while (ConsoleQueue.Count > CONSOLE_QUEUE_CAPACITY)
            {
                ConsoleQueue.TryDequeue(out var _);
            }
            ConsoleQueue.Enqueue(value);
            this.RaisePropertyChanged(nameof(ConsoleText));
        }

        public async void CheckAndExtractBackend()
        {
            await Task.Run(() =>
            {
                var backendArchivePath = Path.GetFullPath("./chaiNNer.7z");

                if (File.Exists(backendArchivePath))
                {
                    IsExtractingBackend = true;
                    using ArchiveFile archiveFile = new(backendArchivePath);
                    archiveFile.Extract(".");
                    archiveFile.Dispose();
                    File.Delete(backendArchivePath);
                    IsExtractingBackend = false;
                }
            });

            DeviceList = await InitializeDeviceList();

            if (DeviceList.Length == 0)
            {
                UseCpu = true;
            }
        }

        public async Task CheckForUpdates()
        {
            try
            {
                if (IsInstalled)
                {
                    await Task.Run(async () =>
                    {
                        _update = await _um.CheckForUpdatesAsync().ConfigureAwait(true);
                    });

                    UpdateStatus();

                    if (AutoUpdateEnabled)
                    {
                        await DownloadUpdate();
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateStatusText = $"Check for update failed: {ex.Message}";
            }
        }

        public async Task DownloadUpdate()
        {
            try
            {
                if (_update != null)
                {
                    ShowDownloadButton = false;
                    await _um.DownloadUpdatesAsync(_update, Progress).ConfigureAwait(true);
                    UpdateStatus();
                }
            }
            catch
            {

            }
        }

        public void ApplyUpdate()
        {
            if (_update != null)
            {
                ShowApplyButton = false;
                _um.ApplyUpdatesAndRestart(_update);
            }
        }

        private void UpdateStatus()
        {
            ShowDownloadButton = false;
            ShowApplyButton = false;
            ShowCheckUpdateButton = true;

            if (_update != null)
            {
                UpdateStatusText = $"Update is available: {_update.TargetFullRelease.Version}";
                ShowDownloadButton = true;
                ShowCheckUpdateButton = false;

                if (_um.IsUpdatePendingRestart)
                {
                    UpdateStatusText = $"Update ready, pending restart to install version: {_update.TargetFullRelease.Version}";
                    ShowDownloadButton = false;
                    ShowApplyButton = true;
                    ShowCheckUpdateButton = false;
                }
                else
                {
                }
            }
            else
            {
                UpdateStatusText = "No updates found";
            }
        }

        private void Progress(int percent)
        {
            UpdateStatusText = $"Downloading update {_update?.TargetFullRelease.Version} ({percent}%)...";
        }


#pragma warning disable CA1822 // Mark members as static
        public async void OpenModelsDirectory()
#pragma warning restore CA1822 // Mark members as static
        {
            await Task.Run(() =>
            {
                Process.Start("explorer.exe", UpscaleChain.PthPath);
            });
        }
    }

    [DataContract]
    public class UpscaleWorkflow : ReactiveObject
    {
        public UpscaleWorkflow()
        {
            var g1 = this.WhenAnyValue
            (
                x => x.InputFilePath,
                x => x.OutputFilename,
                x => x.InputFolderPath,
                x => x.OutputFolderPath
            );

            var g2 = this.WhenAnyValue
            (
                x => x.SelectedTabIndex,
                x => x.UpscaleImages,
                x => x.UpscaleArchives,
                x => x.OverwriteExistingFiles,
                x => x.WebpSelected,
                x => x.PngSelected,
                x => x.JpegSelected
            );

            g1.CombineLatest(g2).Subscribe(x =>
            {
                Validate();
            });

            this.WhenAnyValue(x => x.Vm).Subscribe(x =>
            {
                sub?.Dispose();
                sub = Vm.WhenAnyValue(
                    x => x.SelectedWorkflowIndex,
                    x => x.RequestShowAppSettings
                    ).Subscribe(x =>
                    {
                        this.RaisePropertyChanged(nameof(ActiveWorkflow));
                        Vm?.RaisePropertyChanged("Workflows");
                    });
            });
        }

        private IDisposable? sub;

        private MainWindowViewModel? _vm;
        public MainWindowViewModel? Vm
        {
            get => _vm;
            set => this.RaiseAndSetIfChanged(ref _vm, value);
        }

        private string _workflowName;
        [DataMember]
        public string WorkflowName
        {
            get => _workflowName;
            set => this.RaiseAndSetIfChanged( ref _workflowName, value );
        }

        private int _workflowIndex;
        [DataMember]
        public int WorkflowIndex
        {
            get => _workflowIndex;
            set => this.RaiseAndSetIfChanged(ref _workflowIndex, value);
            
        }

        public string WorkflowIcon => $"Numeric{WorkflowIndex}Circle";

        public bool ActiveWorkflow {
            get
            {
                Debug.WriteLine($"ActiveWorkflow {WorkflowIndex} == {Vm?.SelectedWorkflowIndex}; {Vm == null}");
                return WorkflowIndex == Vm?.SelectedWorkflowIndex && (!Vm?.ShowAppSettings ?? false);
            }
            
        }

        public bool IsDefaultWorkflow => WorkflowIndex == 0;

        private int _selectedTabIndex;
        [DataMember]
        public int SelectedTabIndex
        {
            get => _selectedTabIndex;
            set
            {
                if (_selectedTabIndex != value)
                {
                    this.RaiseAndSetIfChanged(ref _selectedTabIndex, value);
                    //this.RaisePropertyChanged(nameof(InputStatusText));  // TODO

                }
            }
        }

        private string _inputFilePath = string.Empty;
        [DataMember]
        public string InputFilePath
        {
            get => _inputFilePath;
            set
            {
                this.RaiseAndSetIfChanged(ref _inputFilePath, value);
                //this.RaisePropertyChanged(nameof(InputStatusText));  // TODO
            }
        }

        private string _inputFolderPath = string.Empty;
        [DataMember]
        public string InputFolderPath
        {
            get => _inputFolderPath;
            set
            {
                this.RaiseAndSetIfChanged(ref _inputFolderPath, value);
                //this.RaisePropertyChanged(nameof(InputStatusText)); // TODO
            }
        }

        private string _outputFilename = "%filename%-mangajanai";
        [DataMember]
        public string OutputFilename
        {
            get => _outputFilename;
            set => this.RaiseAndSetIfChanged(ref _outputFilename, value);
        }

        private string _outputFolderPath = string.Empty;
        [DataMember]
        public string OutputFolderPath
        {
            get => _outputFolderPath;
            set => this.RaiseAndSetIfChanged(ref _outputFolderPath, value);
        }

        private bool _overwriteExistingFiles = false;
        [DataMember]
        public bool OverwriteExistingFiles
        {
            get => _overwriteExistingFiles;
            set => this.RaiseAndSetIfChanged(ref _overwriteExistingFiles, value);
        }

        private bool _upscaleImages = false;
        [DataMember]
        public bool UpscaleImages
        {
            get => _upscaleImages;
            set => this.RaiseAndSetIfChanged(ref _upscaleImages, value);
        }

        private bool _upscaleArchives = true;
        [DataMember]
        public bool UpscaleArchives
        {
            get => _upscaleArchives;
            set => this.RaiseAndSetIfChanged(ref _upscaleArchives, value);
        }

        private int? _resizeHeightAfterUpscale = 2160;
        [DataMember]
        public int? ResizeHeightAfterUpscale
        {
            get => _resizeHeightAfterUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeHeightAfterUpscale, value ?? 2160);
        }

        private int? _resizeWidthAfterUpscale = 3840;
        [DataMember]
        public int? ResizeWidthAfterUpscale
        {
            get => _resizeWidthAfterUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeWidthAfterUpscale, value ?? 3840);
        }

        private bool _webpSelected = true;
        [DataMember]
        public bool WebpSelected
        {
            get => _webpSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _webpSelected, value);
                this.RaisePropertyChanged(nameof(ShowUseLosslessCompression));
                this.RaisePropertyChanged(nameof(ShowLossyCompressionQuality));
            }
        }

        private bool _pngSelected = false;
        [DataMember]
        public bool PngSelected
        {
            get => _pngSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _pngSelected, value);
            }
        }

        private bool _jpegSelected = false;
        [DataMember]
        public bool JpegSelected
        {
            get => _jpegSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _jpegSelected, value);
                this.RaisePropertyChanged(nameof(ShowLossyCompressionQuality));
            }
        }

        public string ImageFormat => WebpSelected ? "webp" : PngSelected ? "png" : "jpg";

        public bool ShowUseLosslessCompression => WebpSelected;

        private bool _useLosslessCompression = false;
        [DataMember]
        public bool UseLosslessCompression
        {
            get => _useLosslessCompression;
            set
            {
                this.RaiseAndSetIfChanged(ref _useLosslessCompression, value);
                this.RaisePropertyChanged(nameof(ShowLossyCompressionQuality));
            }
        }

        public bool ShowLossyCompressionQuality => JpegSelected || (WebpSelected && !UseLosslessCompression);

        private int? _lossyCompressionQuality = 80;
        [DataMember]
        public int? LossyCompressionQuality
        {
            get => _lossyCompressionQuality;
            set => this.RaiseAndSetIfChanged(ref _lossyCompressionQuality, value ?? 80);
        }

        private bool _showLossySettings = true;
        [DataMember]
        public bool ShowLossySettings
        {
            get => _showLossySettings;
            set => this.RaiseAndSetIfChanged(ref _showLossySettings, value);
        }

        private bool _modeScaleSelected = true;
        [DataMember]
        public bool ModeScaleSelected
        {
            get => _modeScaleSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _modeScaleSelected, value);
            }
        }

        private int _upscaleScaleFactor = 4;
        [DataMember]
        public int UpscaleScaleFactor
        {
            get => _upscaleScaleFactor;
            set 
            { 
                this.RaiseAndSetIfChanged(ref _upscaleScaleFactor, value); 
                this.RaisePropertyChanged(nameof(Is1x));
                this.RaisePropertyChanged(nameof(Is2x));
                this.RaisePropertyChanged(nameof(Is3x));
                this.RaisePropertyChanged(nameof(Is4x));
            }
        }

        public bool Is1x => UpscaleScaleFactor == 1;
        public bool Is2x => UpscaleScaleFactor == 2;
        public bool Is3x => UpscaleScaleFactor == 3;
        public bool Is4x => UpscaleScaleFactor == 4;


        public void SetUpscaleScaleFactor(int scaleFactor)
        {
            UpscaleScaleFactor = scaleFactor;
        }

        private bool _modeWidthSelected = false;
        [DataMember]
        public bool ModeWidthSelected
        {
            get => _modeWidthSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _modeWidthSelected, value);
            }
        }

        private bool _modeHeightSelected = false;
        [DataMember]
        public bool ModeHeightSelected
        {
            get => _modeHeightSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _modeHeightSelected, value);
            }
        }

        private bool _showAdvancedSettings = false;
        [DataMember]
        public bool ShowAdvancedSettings
        {
            get => _showAdvancedSettings;
            set => this.RaiseAndSetIfChanged(ref _showAdvancedSettings, value);
        }

        private AvaloniaList<UpscaleChain> _chains;
        [DataMember]
        public AvaloniaList<UpscaleChain> Chains
        {
            get => _chains;
            set => this.RaiseAndSetIfChanged(ref _chains, value);
        }

        private bool _valid = false;
        [IgnoreDataMember]
        public bool Valid
        {
            get => _valid;
            set
            {
                this.RaiseAndSetIfChanged(ref _valid, value);
                //this.RaisePropertyChanged(nameof(UpscaleEnabled));  // TODO
                //this.RaisePropertyChanged(nameof(LeftStatus));  // TODO
            }
        }

        public void SetWebpSelected()
        {
            WebpSelected = true;
            PngSelected = false;
            JpegSelected = false;
        }

        public void SetPngSelected()
        {
            PngSelected = true;
            WebpSelected = false;
            JpegSelected = false;
        }

        public void SetJpegSelected()
        {
            JpegSelected = true;
            WebpSelected = false;
            PngSelected = false;
        }

        public void SetModeScaleSelected()
        {
            ModeScaleSelected = true;
            ModeWidthSelected = false;
            ModeHeightSelected = false;
        }

        public void SetModeWidthSelected()
        {
            ModeWidthSelected = true;
            ModeScaleSelected = false;
            ModeHeightSelected = false;
        }

        public void SetModeHeightSelected()
        {
            ModeHeightSelected = true;
            ModeScaleSelected = false;
            ModeWidthSelected = false;
        }

        public void Validate()
        {
            var valid = true;
            var validationText = new List<string>();
            if (SelectedTabIndex == 0)
            {

                if (string.IsNullOrWhiteSpace(InputFilePath))
                {
                    valid = false;
                    validationText.Add("Input File is required.");
                }
                else if (!File.Exists(InputFilePath))
                {
                    valid = false;
                    validationText.Add("Input File does not exist.");
                }

            }
            else
            {
                if (string.IsNullOrWhiteSpace(InputFolderPath))
                {
                    valid = false;
                    validationText.Add("Input Folder is required.");
                }
                else if (!Directory.Exists(InputFolderPath))
                {
                    valid = false;
                    validationText.Add("Input Folder does not exist.");
                }
            }

            if (string.IsNullOrWhiteSpace(OutputFilename))
            {
                valid = false;
                validationText.Add("Output Filename is required.");
            }

            if (string.IsNullOrWhiteSpace(OutputFolderPath))
            {
                valid = false;
                validationText.Add("Output Folder is required.");
            }

            Valid = valid;
            // TODO
            //CheckInputs();
            //if (ProgressTotalFiles == 0)
            //{
            //    Valid = false;
            //    validationText.Add($"{InputStatusText} selected for upscaling. At least one file must be selected.");
            //}
            //ValidationText = string.Join("\n", validationText);
        }
    }

    [DataContract]
    public class UpscaleChain : ReactiveObject
    {

        private string _chainNumber = string.Empty;
        [DataMember]
        public string ChainNumber
        {
            get => _chainNumber;
            set => this.RaiseAndSetIfChanged(ref _chainNumber, value);
        }

        private string _minResolution = "0x0";
        [DataMember]
        public string MinResolution
        {
            get => _minResolution;
            set => this.RaiseAndSetIfChanged(ref _minResolution, value);
        }

        private string _maxResolution = "0x0";
        [DataMember]
        public string MaxResolution
        {
            get => _maxResolution;
            set => this.RaiseAndSetIfChanged(ref _maxResolution, value);
        }

        private bool _isGrayscale = false;
        [DataMember]
        public bool IsGrayscale
        {
            get => _isGrayscale;
            set => this.RaiseAndSetIfChanged(ref _isGrayscale, value);
        }

        private bool _isColor = false;
        [DataMember]
        public bool IsColor
        {
            get => _isColor;
            set => this.RaiseAndSetIfChanged(ref _isColor, value);
        }

        private int? _minScaleFactor = 0;
        [DataMember]
        public int? MinScaleFactor
        {
            get => _minScaleFactor;
            set => this.RaiseAndSetIfChanged(ref _minScaleFactor, value ?? 0);
        }

        private int? _maxScaleFactor = 0;
        [DataMember]
        public int? MaxScaleFactor
        {
            get => _maxScaleFactor;
            set => this.RaiseAndSetIfChanged(ref _maxScaleFactor, value ?? 0);
        }

        private string _modelFilePath = string.Empty;
        [DataMember]
        public string ModelFilePath
        {
            get => _modelFilePath;
            set => this.RaiseAndSetIfChanged(ref _modelFilePath, value);
        }

        private string _modelTileSize = "Auto (Estimate)";
        [DataMember]
        public string ModelTileSize
        {
            get => _modelTileSize;
            set => this.RaiseAndSetIfChanged(ref _modelTileSize, value);
        }

        private bool _autoAdjustLevels = false;
        [DataMember]
        public bool AutoAdjustLevels
        {
            get => _autoAdjustLevels;
            set => this.RaiseAndSetIfChanged(ref _autoAdjustLevels, value);
        }

        private int? _resizeHeightBeforeUpscale = 0;
        [DataMember]
        public int? ResizeHeightBeforeUpscale
        {
            get => _resizeHeightBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeHeightBeforeUpscale, value ?? 0);
        }

        private int? _resizeWidthBeforeUpscale = 0;
        [DataMember]
        public int? ResizeWidthBeforeUpscale
        {
            get => _resizeWidthBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeWidthBeforeUpscale, value ?? 0);
        }

        private double? _resizeFactorBeforeUpscale = 100;
        [DataMember]
        public double? ResizeFactorBeforeUpscale
        {
            get => _resizeFactorBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeFactorBeforeUpscale, value ?? 100);
        }


        public static AvaloniaList<string> AllModels => GetAllModels();



        private string[] _tileSizes = [
    "Auto (Estimate)",
            "Maximum",
            "No Tiling",
            "128",
            "192",
            "256",
            "384",
            "512",
            "768",
            "1024",
            "2048",
            "4096"];

        public string[] TileSizes
        {
            get => _tileSizes;
            set => this.RaiseAndSetIfChanged(ref _tileSizes, value);
        }

        public static string PthPath => Path.GetFullPath(@".\chaiNNer\models");

        public static AvaloniaList<string> GetAllModels()
        {
            return new AvaloniaList<string>(Directory.GetFiles(PthPath).Where(filename => Path.GetExtension(filename).Equals(".pth", StringComparison.CurrentCultureIgnoreCase))
                .Select(filename => Path.GetFileNameWithoutExtension(filename))
                .Order().ToList());
        }
    }
}
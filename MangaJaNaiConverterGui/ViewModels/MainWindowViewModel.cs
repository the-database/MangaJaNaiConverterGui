using Avalonia.Data;
using Avalonia.Threading;
using Newtonsoft.Json;
using Progression.Extras;
using ReactiveUI;
using SevenZipExtractor;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.ViewModels
{
    [DataContract]
    public class MainWindowViewModel : ViewModelBase
    {
        public static readonly List<string> IMAGE_EXTENSIONS = new() { ".png", ".jpg", ".jpeg", ".webp", ".bmp" };
        public static readonly List<string> ARCHIVE_EXTENSIONS = new() { ".zip", ".cbz", ".rar", ".cbr"};

        private readonly DispatcherTimer _timer = new ();

        public MainWindowViewModel() 
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

            _timer.Interval = TimeSpan.FromSeconds(1);
            _timer.Tick += _timer_Tick;

            ShowDialog = new Interaction<MainWindowViewModel, MainWindowViewModel?>();
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

        public TimeSpan TotalEtr => _totalEtaCalculator.ETAIsAvailable ? _totalEtaCalculator.ETR : ArchiveEtr + (ElapsedTime + ArchiveEtr)  * (ProgressTotalFiles - (ProgressCurrentFile + 1));

        public string TotalEta => _totalEtaCalculator.ETAIsAvailable ? _totalEtaCalculator.ETA.ToString("t") : _archiveEtaCalculator.ETAIsAvailable ? DateTime.Now.Add(TotalEtr).ToString("t") : "please wait";

        public string AppVersion => Assembly.GetExecutingAssembly().GetName().Version.ToString(3);

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
                    this.RaisePropertyChanged(nameof(InputStatusText));
                    
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
                this.RaisePropertyChanged(nameof(InputStatusText));
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
                this.RaisePropertyChanged(nameof(InputStatusText));
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

        private bool _autoAdjustLevels = false;
        [DataMember]
        public bool AutoAdjustLevels
        {
            get => _autoAdjustLevels;
            set => this.RaiseAndSetIfChanged(ref _autoAdjustLevels, value);
        }

        private string _grayscaleModelFilePath = string.Empty;
        [DataMember]
        public string GrayscaleModelFilePath
        {
            get => _grayscaleModelFilePath;
            set => this.RaiseAndSetIfChanged(ref _grayscaleModelFilePath, value);
        }

        private string _grayscaleModelTileSize = "Auto (Estimate)";
        [DataMember]
        public string GrayscaleModelTileSize
        {
            get => _grayscaleModelTileSize;
            set => this.RaiseAndSetIfChanged(ref _grayscaleModelTileSize, value);
        }

        private string _colorModelFilePath = string.Empty;
        [DataMember]
        public string ColorModelFilePath
        {
            get => _colorModelFilePath;
            set => this.RaiseAndSetIfChanged(ref _colorModelFilePath, value);
        }

        private string _colorModelTileSize = "Auto (Estimate)";
        [DataMember]
        public string ColorModelTileSize
        {
            get => _colorModelTileSize;
            set => this.RaiseAndSetIfChanged(ref _colorModelTileSize, value);
        }

        private string _resizeHeightBeforeUpscale = 0.ToString();
        [DataMember]
        public string ResizeHeightBeforeUpscale
        {
            get => _resizeHeightBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeHeightBeforeUpscale, value);
        }

        private string _resizeWidthBeforeUpscale = 0.ToString();
        [DataMember]
        public string ResizeWidthBeforeUpscale
        {
            get => _resizeWidthBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeWidthBeforeUpscale, value);
        }

        private string _resizeFactorBeforeUpscale = 100.ToString();
        [DataMember]
        public string ResizeFactorBeforeUpscale
        {
            get => _resizeFactorBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeFactorBeforeUpscale, value);
        }

        private string _resizeHeightAfterUpscale = 0.ToString();
        [DataMember]
        public string ResizeHeightAfterUpscale
        {
            get => _resizeHeightAfterUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeHeightAfterUpscale, value);
        }

        private string _resizeWidthAfterUpscale = 0.ToString();
        [DataMember]
        public string ResizeWidthAfterUpscale
        {
            get => _resizeWidthAfterUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeWidthAfterUpscale, value);
        }

        private string _resizeFactorAfterUpscale = 100.ToString();
        [DataMember]
        public string ResizeFactorAfterUpscale
        {
            get => _resizeFactorAfterUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeFactorAfterUpscale, value);
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

        private string ImageFormat => WebpSelected ? "webp" : PngSelected ? "png" : "jpg";

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

        private string _lossyCompressionQuality = 80.ToString();
        [DataMember]
        public string LossyCompressionQuality
        {
            get => _lossyCompressionQuality;
            set => this.RaiseAndSetIfChanged(ref _lossyCompressionQuality, value);
        }

        private bool _showLossySettings = true;
        [DataMember]
        public bool ShowLossySettings
        {
            get => _showLossySettings;
            set => this.RaiseAndSetIfChanged(ref _showLossySettings, value);
        }

        private bool _showAdvancedSettings = false;
        [DataMember]
        public bool ShowAdvancedSettings
        {
            get => _showAdvancedSettings;
            set => this.RaiseAndSetIfChanged(ref _showAdvancedSettings, value);
        }

        private bool _valid = false;
        [IgnoreDataMember]
        public bool Valid
        {
            get => _valid;
            set
            {
                this.RaiseAndSetIfChanged(ref _valid, value);
                this.RaisePropertyChanged(nameof(UpscaleEnabled));
                this.RaisePropertyChanged(nameof(LeftStatus));
            }
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

        public string LeftStatus => !Valid ? ValidationText.Replace("\n", " ") : $"{InputStatusText} selected for upscaling.";

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

        public bool UpscaleEnabled => Valid && !Upscaling;

        private TimeSpan _elapsedTime = TimeSpan.FromSeconds(0);
        public TimeSpan ElapsedTime
        {
            get => _elapsedTime;
            set
            {
                this.RaiseAndSetIfChanged(ref _elapsedTime, value);
            }
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
                if (UpscaleArchives)
                {
                    flags.Append("--upscale-archives ");
                }
                if (UpscaleImages)
                {
                    flags.Append("--upscale-images ");
                }
                if (OverwriteExistingFiles)
                {
                    flags.Append("--overwrite-existing-files ");
                }
                if (AutoAdjustLevels)
                {
                    flags.Append("--auto-adjust-levels ");
                }
                if (UseLosslessCompression)
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

                var inputArgs = $"--input-file-path \"{InputFilePath}\" ";

                if (SelectedTabIndex == 1)
                {
                    inputArgs = $"--input-folder-path \"{InputFolderPath}\" ";
                }

                var grayscaleModelFilePath = string.IsNullOrWhiteSpace(GrayscaleModelFilePath) ? GrayscaleModelFilePath : Path.GetFullPath(GrayscaleModelFilePath);
                var colorModelFilePath = string.IsNullOrWhiteSpace(ColorModelFilePath) ? ColorModelFilePath : Path.GetFullPath(ColorModelFilePath);

                var cmd = $@".\python\python.exe "".\backend\src\runmangajanaiconverterguiupscale.py"" --selected-device {SelectedDeviceIndex} {inputArgs} --output-folder-path ""{OutputFolderPath}"" --output-filename ""{OutputFilename}"" --resize-height-before-upscale {ResizeHeightBeforeUpscale} --resize-width-before-upscale {ResizeWidthBeforeUpscale} --resize-factor-before-upscale {ResizeFactorBeforeUpscale} --grayscale-model-path ""{grayscaleModelFilePath}"" --grayscale-model-tile-size ""{GrayscaleModelTileSize}"" --color-model-path ""{colorModelFilePath}"" --color-model-tile-size ""{ColorModelTileSize}"" --image-format {ImageFormat} --lossy-compression-quality {LossyCompressionQuality} --resize-height-after-upscale {ResizeHeightAfterUpscale} --resize-width-after-upscale {ResizeWidthAfterUpscale} --resize-factor-after-upscale {ResizeFactorAfterUpscale} {flags}";
                ConsoleQueueEnqueue($"Upscaling with command: {cmd}");
                await RunCommand($@" /C {cmd}");

                Valid = true;
            }, ct);

            try
            {
                _timer.Start();
                await task;
                _timer.Stop();
                Validate();
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
                Validate();
            }
            catch { }
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

        private void CheckInputs()
        {
            if (Valid && !Upscaling)
            {
                var overwriteText = OverwriteExistingFiles ? "overwritten" : "skipped";

                // input file
                if (SelectedTabIndex == 0)
                {
                    StringBuilder status = new();
                    var skipFiles = 0;

                    

                    if (IMAGE_EXTENSIONS.Any(x => InputFilePath.ToLower().EndsWith(x))) 
                    {
                        var outputFilePath = Path.ChangeExtension(
                                                Path.Join(
                                                    Path.GetFullPath(OutputFolderPath), 
                                                    OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(InputFilePath))), 
                                                ImageFormat);
                        if (File.Exists(outputFilePath)) {
                            status.Append($" (1 image already exists and will be {overwriteText})");
                            if (!OverwriteExistingFiles)
                            {
                                skipFiles++;
                            }
                        }
                    }
                    else if (ARCHIVE_EXTENSIONS.Any(x => InputFilePath.ToLower().EndsWith(x)))
                    {
                        var outputFilePath = Path.ChangeExtension(
                                                Path.Join(
                                                    Path.GetFullPath(OutputFolderPath),
                                                    OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(InputFilePath))),
                                                "cbz");
                        if (File.Exists(outputFilePath))
                        {
                            status.Append($" (1 archive already exists and will be {overwriteText})");
                            if (!OverwriteExistingFiles)
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
                    if (IMAGE_EXTENSIONS.Any(x => InputFilePath.ToLower().EndsWith(x)))
                    {
                        status.Insert(0, $"{1 - skipFiles} image{s}");
                    }
                    else if (ARCHIVE_EXTENSIONS.Any(x => InputFilePath.ToLower().EndsWith(x)))
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

                    if (UpscaleImages)
                    {
                        var images = Directory.EnumerateFiles(InputFolderPath, "*.*", SearchOption.AllDirectories)
                            .Where(file => IMAGE_EXTENSIONS.Any(ext => file.ToLower().EndsWith(ext)));
                        var imagesCount = 0;

                        foreach (var inputImagePath in images)
                        {
                            var outputImagePath = Path.ChangeExtension(
                                                    Path.Join(
                                                        Path.GetFullPath(OutputFolderPath),
                                                        OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputImagePath))),
                                                    ImageFormat);
                            // if out file exists, exist count ++
                            // if overwrite image OR out file doesn't exist, count image++
                            var fileExists = File.Exists(outputImagePath);

                            if (fileExists)
                            {
                                existImageCount++;
                            }

                            if (!fileExists || OverwriteExistingFiles)
                            {
                                imagesCount++;
                            }
                        }

                        var imageS = imagesCount == 1 ? "" : "s";
                        var existImageS = existImageCount == 1 ? "" : "s";

                        statuses.Add($"{imagesCount} image{imageS} ({existImageCount} image{existImageS} already exist and will be {overwriteText})");
                        totalFileCount += imagesCount;
                    }
                    if (UpscaleArchives)
                    {
                        var archives = Directory.EnumerateFiles(InputFolderPath, "*.*", SearchOption.AllDirectories)
                            .Where(file => ARCHIVE_EXTENSIONS.Any(ext => file.ToLower().EndsWith(ext)));
                        var archivesCount = 0;

                        foreach (var inputArchivePath in archives)
                        {
                            var outputArchivePath = Path.ChangeExtension(
                                                        Path.Join(
                                                            Path.GetFullPath(OutputFolderPath),
                                                            OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputArchivePath))),
                                                        "cbz");
                            var fileExists = File.Exists(outputArchivePath); 

                            if (fileExists)
                            {
                                existArchiveCount++;
                            }

                            if (!fileExists || OverwriteExistingFiles)
                            {
                                archivesCount++;
                            }
                        }

                        var archiveS = archivesCount == 1 ? "" : "s";
                        var existArchiveS = existArchiveCount == 1 ? "" : "s";
                        statuses.Add($"{archivesCount} archive{archiveS} ({existArchiveCount} archive{existArchiveS} already exist and will be {overwriteText})");
                        totalFileCount += archivesCount;
                    }

                    if (!UpscaleArchives && !UpscaleImages)
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
            CheckInputs();
            if (ProgressTotalFiles == 0)
            {
                Valid = false;
                validationText.Add($"{InputStatusText} selected for upscaling. At least one file must be selected.");
            }
            ValidationText = string.Join("\n", validationText);
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
                using (var outputFile = new StreamWriter("error.log", append: true))
                {
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
    }
}
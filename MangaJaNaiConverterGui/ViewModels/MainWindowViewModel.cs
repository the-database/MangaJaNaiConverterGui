using Avalonia.Collections;
using Avalonia.Threading;
using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.Services;
using Newtonsoft.Json;
using ReactiveUI;
using Splat;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reactive.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Velopack;
using File = System.IO.File;
using Path = System.IO.Path;

namespace MangaJaNaiConverterGui.ViewModels
{
    [DataContract]
    public class MainWindowViewModel : ViewModelBase
    {
        public static readonly List<string> IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"];
        public static readonly List<string> ARCHIVE_EXTENSIONS = [".zip", ".cbz", ".rar", ".cbr"];

        private readonly DispatcherTimer _timer = new();
        private static readonly HttpClient client = new();

        private UpdateInfo? _update = null;

        private readonly IPythonService _pythonService;
        private readonly IUpdateManagerService _updateManagerService;
        private readonly ISuspensionDriverService _suspensionDriverService;

        public MainWindowViewModel(IPythonService? pythonService = null, IUpdateManagerService? updateManagerService = null, ISuspensionDriverService? suspensionDriverService = null)
        {
            _pythonService = pythonService ?? Locator.Current.GetService<IPythonService>()!;
            _updateManagerService = updateManagerService ?? Locator.Current.GetService<IUpdateManagerService>()!;
            _suspensionDriverService = suspensionDriverService ?? Locator.Current.GetService<ISuspensionDriverService>()!;

            var g1 = this.WhenAnyValue
            (
                x => x.SelectedWorkflowIndex
            ).Subscribe(x =>
            {
                CurrentWorkflow?.Validate();
            });

            _timer.Interval = TimeSpan.FromSeconds(1);
            _timer.Tick += _timer_Tick;

            ShowDialog = new Interaction<MainWindowViewModel, MainWindowViewModel?>();

            CheckAndDoBackup();
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

        private static readonly string DEFAULT_WORKFLOW = """
{
  "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleWorkflow, MangaJaNaiConverterGui",
  "WorkflowName": "Upscale Manga (Default)",
  "WorkflowIndex": 0,
  "SelectedTabIndex": 0,
  "InputFilePath": "",
  "InputFolderPath": "",
  "OutputFilename": "%filename%-mangajanai",
  "OutputFolderPath": "",
  "OverwriteExistingFiles": false,
  "UpscaleImages": true,
  "UpscaleArchives": true,
  "ResizeHeightAfterUpscale": 2160,
  "ResizeWidthAfterUpscale": 3840,
  "WebpSelected": true,
  "AvifSelected": false,
  "PngSelected": false,
  "JpegSelected": false,
  "UseLosslessCompression": false,
  "LossyCompressionQuality": 80,
  "ShowLossySettings": true,
  "ModeScaleSelected": true,
  "UpscaleScaleFactor": 4,
  "ModeWidthSelected": false,
  "ModeHeightSelected": false,
  "ModeFitToDisplaySelected": false,
  "DisplayDevice": "Kobo Elipsa 2E (2023)",
  "DisplayDeviceWidth": 1404,
  "DisplayDeviceHeight": 1872,
  "DisplayPortraitSelected": true,
  "ShowAdvancedSettings": false,
  "GrayscaleDetectionThreshold": 12,
  "Chains": {
    "$type": "Avalonia.Collections.AvaloniaList`1[[MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui]], Avalonia.Base",
    "$values": [
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "1",
        "MinResolution": "0x0",
        "MaxResolution": "0x0",
        "IsGrayscale": false,
        "IsColor": true,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": false,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "2",
        "MinResolution": "0x0",
        "MaxResolution": "0x0",
        "IsGrayscale": false,
        "IsColor": true,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": false,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "3",
        "MinResolution": "0x0",
        "MaxResolution": "0x1250",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "4",
        "MinResolution": "0x0",
        "MaxResolution": "0x1250",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_1200p_V1_ESRGAN_70k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "5",
        "MinResolution": "0x1251",
        "MaxResolution": "0x1350",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_1300p_V1_ESRGAN_75k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "6",
        "MinResolution": "0x1251",
        "MaxResolution": "0x1350",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_1300p_V1_ESRGAN_75k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "7",
        "MinResolution": "0x1351",
        "MaxResolution": "0x1450",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_1400p_V1_ESRGAN_70k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "8",
        "MinResolution": "0x1351",
        "MaxResolution": "0x1450",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_1400p_V1_ESRGAN_105k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "9",
        "MinResolution": "0x1451",
        "MaxResolution": "0x1550",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_1500p_V1_ESRGAN_90k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "10",
        "MinResolution": "0x1451",
        "MaxResolution": "0x1550",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_1500p_V1_ESRGAN_105k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "11",
        "MinResolution": "0x1551",
        "MaxResolution": "0x1760",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_1600p_V1_ESRGAN_90k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "12",
        "MinResolution": "0x1551",
        "MaxResolution": "0x1760",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_1600p_V1_ESRGAN_70k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "13",
        "MinResolution": "0x1761",
        "MaxResolution": "0x1984",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "14",
        "MinResolution": "0x1761",
        "MaxResolution": "0x1984",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_1920p_V1_ESRGAN_105k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "15",
        "MinResolution": "0x1985",
        "MaxResolution": "0x0",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 0,
        "MaxScaleFactor": 2,
        "ModelFilePath": "2x_MangaJaNai_2048p_V1_ESRGAN_95k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      },
      {
        "$type": "MangaJaNaiConverterGui.ViewModels.UpscaleChain, MangaJaNaiConverterGui",
        "ChainNumber": "16",
        "MinResolution": "0x1985",
        "MaxResolution": "0x0",
        "IsGrayscale": true,
        "IsColor": false,
        "MinScaleFactor": 2,
        "MaxScaleFactor": 0,
        "ModelFilePath": "4x_MangaJaNai_2048p_V1_ESRGAN_70k.pth",
        "ModelTileSize": "Auto (Estimate)",
        "AutoAdjustLevels": true,
        "ResizeHeightBeforeUpscale": 0,
        "ResizeWidthBeforeUpscale": 0,
        "ResizeFactorBeforeUpscale": 100.0
      }
    ]
  }
}
""";

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

        public bool IsInstalled => _updateManagerService.IsInstalled;
        [DataMember]
        public string ModelsDirectory => _pythonService.ModelsDirectory;

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

        public string AppVersion => _updateManagerService.AppVersion;

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

        private string _pythonPipList = string.Empty;
        public string PythonPipList
        {
            get => _pythonPipList;
            set => this.RaiseAndSetIfChanged(ref _pythonPipList, value);
        }

        private AvaloniaDictionary<string, ReaderDevice> _displayDeviceMap = [];
        [DataMember]
        public AvaloniaDictionary<string, ReaderDevice> DisplayDeviceMap
        {
            get => _displayDeviceMap;
            set => this.RaiseAndSetIfChanged(ref _displayDeviceMap, value);
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

        private string _backendSetupMainStatus = string.Empty;
        public string BackendSetupMainStatus
        {
            get => this._backendSetupMainStatus;
            set
            {
                this.RaiseAndSetIfChanged(ref _backendSetupMainStatus, value);
            }
        }

        public string BackendSetupSubStatusText => string.Join("\n", BackendSetupSubStatusQueue);

        private static readonly int BACKEND_SETUP_SUB_STATUS_QUEUE_CAPACITY = 50;

        private ConcurrentQueue<string> _backendSetupSubStatusQueue = new();
        public ConcurrentQueue<string> BackendSetupSubStatusQueue
        {
            get => this._backendSetupSubStatusQueue;
            set
            {
                this.RaiseAndSetIfChanged(ref _backendSetupSubStatusQueue, value);
                this.RaisePropertyChanged(nameof(BackendSetupSubStatusText));
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

        public string PythonPath => _pythonService.PythonPath;

        private bool _isExtractingBackend = true;
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

        private AvaloniaList<UpscaleWorkflow>? _workflows;
        [DataMember]
        public AvaloniaList<UpscaleWorkflow>? Workflows
        {
            get => _workflows;
            set => this.RaiseAndSetIfChanged(ref _workflows, value);
        }

        public AvaloniaList<UpscaleWorkflow> CustomWorkflows => new(Workflows.Skip(1).ToList());

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

        public UpscaleWorkflow? CurrentWorkflow
        {
            get => Workflows?[SelectedWorkflowIndex];
            set
            {
                if (Workflows != null)
                {
                    Workflows[SelectedWorkflowIndex] = value;
                    this.RaisePropertyChanged(nameof(CurrentWorkflow));
                    this.RaisePropertyChanged(nameof(CustomWorkflows));
                }
            }
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
                await _suspensionDriverService.SuspensionDriver.SaveState(this);
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

                var cmd = $@".\python\python\python.exe ""{Path.GetFullPath(@".\backend\src\run_upscale.py")}"" --settings ""{_pythonService.AppStatePath}""";
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


        public void CheckInputs()
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
                        var outputFilePath = Path.Join(
                                                    Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                    CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(CurrentWorkflow.InputFilePath))) + $".{CurrentWorkflow.ImageFormat}";
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
                        var outputFilePath = Path.Join(Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                            CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(CurrentWorkflow.InputFilePath))) + ".cbz";

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
                            var outputImagePath = Path.Join(
                                                        Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                        CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputImagePath))) + $"{CurrentWorkflow.ImageFormat}";
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
                            var outputArchivePath = Path.Join(
                                                            Path.GetFullPath(CurrentWorkflow.OutputFolderPath),
                                                            CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputArchivePath))) + ".cbz";
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
            CurrentWorkflow?.Chains.Add(new UpscaleChain
            {
                Vm = this,
            });
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
                process.StartInfo.WorkingDirectory = _pythonService.BackendDirectory;
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;

                // Create a StreamWriter to write the output to a log file
                using (var outputFile = new StreamWriter(Path.Combine(_pythonService.LogsDirectory, "upscale.log"), append: false))
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
            if (!File.Exists(@".\backend\src\device_list.py"))
            {
                return [];
            }

            // Create a new process to run the CMD command
            using (var process = new Process())
            {
                _runningProcess = process;
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.Arguments = @$"/C .\python\python\python.exe {Path.GetFullPath(@".\backend\src\device_list.py")}";
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.WorkingDirectory = _pythonService.BackendDirectory;
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;

                var result = string.Empty;

                // Create a StreamWriter to write the output to a log file
                try
                {
                    using var outputFile = new StreamWriter(Path.Combine(_pythonService.LogsDirectory, "upscale.log"), append: false);
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

        public async Task<string> RunPythonPipList()
        {
            List<string> result = [];

            // Create a new process to run the CMD command
            using (var process = new Process())
            {
                _runningProcess = process;
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.Arguments = @$"/C .\python\python\python.exe -m pip list";
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.WorkingDirectory = _pythonService.BackendDirectory;
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;

                // Create a StreamWriter to write the output to a log file
                try
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
                            result.Add(e.Data);
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();
                }
                catch (IOException) { }
            }

            return string.Join("\n", result);
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

        private void BackendSetupSubStatusQueueEnqueue(string value)
        {
            while (BackendSetupSubStatusQueue.Count > BACKEND_SETUP_SUB_STATUS_QUEUE_CAPACITY)
            {
                BackendSetupSubStatusQueue.TryDequeue(out var _);
            }
            BackendSetupSubStatusQueue.Enqueue(value);
            this.RaisePropertyChanged(nameof(BackendSetupSubStatusText));
        }

        public void ReadWorkflowFileToCurrentWorkflow(string fullPath)
        {
            if (!File.Exists(fullPath))
            {
                return;
            }

            var lines = File.ReadAllText(fullPath);
            var workflow = JsonConvert.DeserializeObject<UpscaleWorkflow>(lines, NewtonsoftJsonSuspensionDriver.Settings);
            if (workflow != null && CurrentWorkflow != null)
            {
                workflow.WorkflowIndex = CurrentWorkflow.WorkflowIndex;
                workflow.Vm = CurrentWorkflow.Vm;
                CurrentWorkflow = workflow;
            }
        }

        public void WriteCurrentWorkflowToFile(string fullPath)
        {
            var lines = JsonConvert.SerializeObject(CurrentWorkflow, NewtonsoftJsonSuspensionDriver.Settings);
            File.WriteAllText(fullPath, lines);
        }

        public async Task CheckAndExtractBackend()
        {
            await Task.Run(async () =>
            {
                IsExtractingBackend = true;

                if (!Directory.Exists(_pythonService.LogsDirectory))
                {
                    Directory.CreateDirectory(_pythonService.LogsDirectory);
                }

                if (!_pythonService.AreModelsInstalled())
                {
                    await DownloadModels();
                }

                if (!_pythonService.IsPythonInstalled() || !(await _pythonService.IsBackendUpdated()))
                {
                    // Download Python tgz
                    BackendSetupMainStatus = "Downloading Python Backend...";
                    var downloadUrl = _pythonService.BackendUrl;
                    var targetPath = Path.Join(_pythonService.PythonDirectory, "backend.7z");
                    if (Directory.Exists(_pythonService.PythonDirectory))
                    {
                        Directory.Delete(_pythonService.PythonDirectory, true);
                    }
                    Directory.CreateDirectory(_pythonService.PythonDirectory);
                    await Downloader.DownloadFileAsync(downloadUrl, targetPath, (progress) =>
                    {
                        BackendSetupMainStatus = $"Downloading Python Backend ({progress}%)...";
                    });

                    // Extract Python 7z
                    BackendSetupMainStatus = "Extracting Python Backend...";
                    _pythonService.Extract7z(targetPath, _pythonService.PythonDirectory);

                    Directory.Move(Path.Combine(_pythonService.PythonDirectory, "backend", "python"), Path.Combine(_pythonService.PythonDirectory, "python"));

                    using (StreamWriter sw = File.CreateText(_pythonService.PythonBackendVersionPath))
                    {
                        sw.WriteLine(_pythonService.BackendVersion);
                    }

                    Directory.Delete(Path.Combine(_pythonService.PythonDirectory, "backend"));
                    File.Delete(targetPath);
                }

                IsExtractingBackend = false;
            });

            DeviceList = await InitializeDeviceList();

            if (DeviceList.Length == 0)
            {
                UseCpu = true;
            }

            PythonPipList = await RunPythonPipList();
        }

        public async Task ReinstallBackend()
        {
            if (Directory.Exists(_pythonService.ModelsDirectory))
            {
                Directory.Delete(_pythonService.ModelsDirectory, true);
            }

            if (Directory.Exists(_pythonService.PythonDirectory))
            {
                Directory.Delete(_pythonService.PythonDirectory, true);
            }

            await CheckAndExtractBackend();
        }

        public async Task DownloadModels()
        {
            BackendSetupMainStatus = "Downloading MangaJaNai Models...";
            var download = "https://github.com/the-database/mangajanai/releases/download/1.0.0/MangaJaNai_V1_ModelsOnly.zip";
            var targetPath = Path.Join(_pythonService.ModelsDirectory, "mangajanai.zip");
            Directory.CreateDirectory(_pythonService.ModelsDirectory);
            await Downloader.DownloadFileAsync(download, targetPath, (progress) =>
            {
                BackendSetupMainStatus = $"Downloading MangaJaNai Models ({progress}%)...";
            });

            BackendSetupMainStatus = "Extracting MangaJaNai Models...";
            _pythonService.ExtractZip(targetPath, _pythonService.ModelsDirectory, (double progress) =>
            {
                BackendSetupMainStatus = $"Extracting MangaJaNai Models ({progress}%)...";
            });
            File.Delete(targetPath);

            BackendSetupMainStatus = "Downloading IllustrationJaNai V3denoise Models...";
            download = "https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3denoise.zip";
            targetPath = Path.Join(_pythonService.ModelsDirectory, "illustrationjanai.zip");
            await Downloader.DownloadFileAsync(download, targetPath, (progress) =>
            {
                BackendSetupMainStatus = $"Downloading IllustrationJaNai V3denoise Models ({progress}%)...";
            });

            BackendSetupMainStatus = "Extracting IllustrationJaNai V3denoise Models...";
            _pythonService.ExtractZip(targetPath, _pythonService.ModelsDirectory, (double progress) =>
            {
                BackendSetupMainStatus = $"Extracting IllustrationJaNai V3denoise Models ({progress}%)...";
            });
            File.Delete(targetPath);

            BackendSetupMainStatus = "Downloading IllustrationJaNai V3detail Models...";
            download = "https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3detail.zip";
            targetPath = Path.Join(_pythonService.ModelsDirectory, "illustrationjanai.zip");
            await Downloader.DownloadFileAsync(download, targetPath, (progress) =>
            {
                BackendSetupMainStatus = $"Downloading IllustrationJaNai V3detail Models ({progress}%)...";
            });

            BackendSetupMainStatus = "Extracting IllustrationJaNai V3detail Models...";
            _pythonService.ExtractZip(targetPath, _pythonService.ModelsDirectory, (double progress) =>
            {
                BackendSetupMainStatus = $"Extracting IllustrationJaNai V3detail Models ({progress}%)...";
            });
            File.Delete(targetPath);
        }



        public async Task<string[]> InstallUpdatePythonDependencies()
        {
            var cmd = _pythonService.InstallUpdatePythonDependenciesCommand;
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
                process.StartInfo.WorkingDirectory = _pythonService.BackendDirectory;

                var result = string.Empty;
                using var outputFile = new StreamWriter(Path.Combine(_pythonService.LogsDirectory, "install.log"));
                outputFile.WriteLine($"Working Directory: {process.StartInfo.WorkingDirectory}");
                outputFile.WriteLine($"Run Command: {cmd}");
                // Create a StreamWriter to write the output to a log file
                try
                {
                    //using var outputFile = new StreamWriter("error.log", append: true);
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            //Debug.WriteLine($"STDERR = {e.Data}");
                            outputFile.WriteLine(e.Data);
                            BackendSetupSubStatusQueueEnqueue(e.Data);
                        }
                    };

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            result = e.Data;
                            outputFile.WriteLine(e.Data);
                            //Debug.WriteLine($"STDOUT = {e.Data}");
                            BackendSetupSubStatusQueueEnqueue(e.Data);
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();
                }
                catch (IOException) { }
            }

            return [];
        }

        public void CheckAndDoBackup()
        {
            Task.Run(() =>
            {
                try
                {
                    if (!File.Exists(_pythonService.AppStatePath))
                        return;

                    var files = Directory.EnumerateFiles(_pythonService.AppStateFolder)
                        .Where(f =>
                        {
                            var name = Path.GetFileName(f);
                            return name.StartsWith("autobackup_") &&
                                   name.EndsWith(_pythonService.AppStateFilename);
                        })
                        .OrderByDescending(f => f)
                        .ToList();

                    var latestBackup = files.FirstOrDefault();
                    if (latestBackup is not null &&
                        FilesAreEqual(_pythonService.AppStatePath, latestBackup))
                    {
                        return;
                    }

                    var backupName =
                        $"autobackup_{DateTime.Now:yyyyMMdd-HHmmss}_{_pythonService.AppStateFilename}";
                    var backupPath = Path.Combine(_pythonService.AppStateFolder, backupName);

                    File.Copy(_pythonService.AppStatePath, backupPath);

                    files.Insert(0, backupPath);

                    const int maxBackups = 10;
                    if (files.Count > maxBackups)
                    {
                        foreach (var old in files.Skip(maxBackups))
                        {
                            try { File.Delete(old); }
                            catch { }
                        }
                    }
                }
                catch
                {
                }
            });
        }

        private static bool FilesAreEqual(string path1, string path2)
        {
            var info1 = new FileInfo(path1);
            var info2 = new FileInfo(path2);
            if (info1.Length != info2.Length)
                return false;

            var bytes1 = File.ReadAllBytes(path1);
            var bytes2 = File.ReadAllBytes(path2);

            return bytes1.AsSpan().SequenceEqual(bytes2);
        }

        public void ResetCurrentWorkflow()
        {
            if (CurrentWorkflow != null)
            {
                var workflow = JsonConvert.DeserializeObject<UpscaleWorkflow>(DEFAULT_WORKFLOW, NewtonsoftJsonSuspensionDriver.Settings);
                var workflowIndex = CurrentWorkflow.WorkflowIndex;
                var workflowName = $"Custom Workflow {workflowIndex}";

                if (workflow != null)
                {
                    var defaultWorkflow = new UpscaleWorkflow
                    {
                        Vm = this,
                        WorkflowIndex = workflowIndex,
                        WorkflowName = workflowName,
                        Chains = workflow.Chains
                    };

                    foreach (var chain in defaultWorkflow.Chains)
                    {
                        chain.Vm = this;
                    }

                    CurrentWorkflow = defaultWorkflow;
                }
            }
        }

        public async Task<IEnumerable<object>> PopulateDevicesAsync(string? searchText, CancellationToken cancellationToken)
        {
            try
            {
                var requestUrl = $"https://animejan.ai/mangajanai/api/search?q={Uri.EscapeDataString(searchText?.Trim() ?? "")}&p=0&s=4";
                if (string.IsNullOrWhiteSpace(searchText))
                {
                    requestUrl = $"https://animejan.ai/mangajanai/api/top";
                }
                var response = await client.GetStringAsync(requestUrl, cancellationToken);
                var devices = JsonConvert.DeserializeObject<List<ReaderDevice>>(response, NewtonsoftJsonSuspensionDriver.Settings);
                if (devices != null)
                {
                    foreach (var device in devices)
                    {
                        DisplayDeviceMap[device.ToString()] = device;

                    }
                    return devices.ToList();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }

            return [];
        }

        public async Task CheckForUpdates()
        {
            try
            {
                if (_updateManagerService.IsInstalled)
                {
                    await Task.Run(async () =>
                    {
                        _update = await _updateManagerService.CheckForUpdatesAsync().ConfigureAwait(true);
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
                    await _updateManagerService.DownloadUpdatesAsync(_update, Progress).ConfigureAwait(true);
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
                _updateManagerService.ApplyUpdatesAndRestart(_update);
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

                if (_updateManagerService.IsUpdatePendingRestart)
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


        public async void OpenModelsDirectory()
        {
            await Task.Run(() =>
            {
                Process.Start("explorer.exe", _pythonService.ModelsDirectory);
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
                x => x.OutputFolderPath,
                x => x.SelectedTabIndex,
                x => x.DisplayDevice,
                x => x.DisplayPortraitSelected
            );

            var g2 = this.WhenAnyValue
            (
                x => x.UpscaleImages,
                x => x.UpscaleArchives,
                x => x.OverwriteExistingFiles,
                x => x.WebpSelected,
                x => x.PngSelected,
                x => x.JpegSelected,
                x => x.AvifSelected
            );

            var g3 = this.WhenAnyValue
            (
                x => x.ModeFitToDisplaySelected,
                x => x.ModeHeightSelected,
                x => x.ModeWidthSelected,
                x => x.ResizeHeightAfterUpscale,
                x => x.ResizeWidthAfterUpscale
            );

            g1.CombineLatest(g2).CombineLatest(g3).Subscribe(x =>
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

            this.WhenAnyValue(x => x.InputFilePath).Subscribe(x =>
            {
                if (string.IsNullOrWhiteSpace(OutputFolderPath) && !string.IsNullOrWhiteSpace(InputFilePath))
                {
                    try
                    {
                        OutputFolderPath = Directory.GetParent(InputFilePath)?.ToString() ?? "";
                    }
                    catch (Exception)
                    {

                    }
                }
            });

            this.WhenAnyValue(x => x.InputFolderPath).Subscribe(x =>
            {
                if (string.IsNullOrWhiteSpace(OutputFolderPath) && !string.IsNullOrWhiteSpace(InputFolderPath))
                {
                    try
                    {
                        OutputFolderPath = $"{InputFolderPath} mangajanai";
                    }
                    catch (Exception)
                    {

                    }
                }
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
            set => this.RaiseAndSetIfChanged(ref _workflowName, value);
        }

        private int _workflowIndex;
        [DataMember]
        public int WorkflowIndex
        {
            get => _workflowIndex;
            set => this.RaiseAndSetIfChanged(ref _workflowIndex, value);

        }

        public string WorkflowIcon => $"Numeric{WorkflowIndex}Circle";

        public bool ActiveWorkflow
        {
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
                    Vm?.RaisePropertyChanged(nameof(Vm.InputStatusText));  // TODO
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
                Vm?.RaisePropertyChanged(nameof(Vm.InputStatusText));  // TODO
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
                Vm?.RaisePropertyChanged(nameof(Vm.InputStatusText)); // TODO
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

        private bool _avifSelected = false;
        [DataMember]
        public bool AvifSelected
        {
            get => _avifSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _avifSelected, value);
                this.RaisePropertyChanged(nameof(ShowLossyCompressionQuality));
                this.RaisePropertyChanged(nameof(ShowUseLosslessCompression));
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

        public string ImageFormat => WebpSelected ? "webp" : PngSelected ? "png" : AvifSelected ? "avif" : "jpg";

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

        public bool ShowLossyCompressionQuality => JpegSelected || (WebpSelected && !UseLosslessCompression) || AvifSelected;

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

        private bool _modeFitToDisplaySelected = false;
        [DataMember]
        public bool ModeFitToDisplaySelected
        {
            get => _modeFitToDisplaySelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _modeFitToDisplaySelected, value);
            }
        }

        private string _displayDevice;
        [DataMember]
        public string DisplayDevice
        {
            get => _displayDevice;
            set
            {
                this.RaiseAndSetIfChanged(ref _displayDevice, value);
                this.RaisePropertyChanged(nameof(DisplayDeviceWidth));
                this.RaisePropertyChanged(nameof(DisplayDeviceHeight));
            }
        }

        [DataMember]
        public int DisplayDeviceWidth
        {
            get
            {
                if (Vm != null && DisplayDevice != null)
                {
                    Vm.DisplayDeviceMap.TryGetValue(DisplayDevice, out var displayDevice);
                    if (displayDevice != null)
                    {
                        return DisplayPortraitSelected ? displayDevice.Width : displayDevice.Height;
                    }
                }

                return 0;
            }
        }

        [DataMember]
        public int DisplayDeviceHeight
        {
            get
            {
                if (Vm != null && DisplayDevice != null)
                {
                    Vm.DisplayDeviceMap.TryGetValue(DisplayDevice, out var displayDevice);
                    if (displayDevice != null)
                    {
                        return DisplayPortraitSelected ? displayDevice.Height : displayDevice.Width;
                    }
                }

                return 0;
            }
        }

        private bool _displayPortraitSelected = true;
        [DataMember]
        public bool DisplayPortraitSelected
        {
            get => _displayPortraitSelected;
            set
            {
                this.RaiseAndSetIfChanged(ref _displayPortraitSelected, value);
                this.RaisePropertyChanged(nameof(DisplayDeviceWidth));
                this.RaisePropertyChanged(nameof(DisplayDeviceHeight));
            }
        }

        private bool _showAdvancedSettings = false;
        [DataMember]
        public bool ShowAdvancedSettings
        {
            get => _showAdvancedSettings;
            set => this.RaiseAndSetIfChanged(ref _showAdvancedSettings, value);
        }

        private int _grayscaleDetectionThreshold = 12;
        [DataMember]
        public int GrayscaleDetectionThreshold
        {
            get => _grayscaleDetectionThreshold;
            set => this.RaiseAndSetIfChanged(ref _grayscaleDetectionThreshold, value);
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
                if (Vm != null)
                {
                    Vm.RaisePropertyChanged(nameof(Vm.UpscaleEnabled));  // TODO
                    Vm.RaisePropertyChanged(nameof(Vm.LeftStatus));  // TODO
                }
            }
        }

        public void SetWebpSelected()
        {
            WebpSelected = true;
            PngSelected = false;
            JpegSelected = false;
            AvifSelected = false;
        }

        public void SetPngSelected()
        {
            PngSelected = true;
            WebpSelected = false;
            JpegSelected = false;
            AvifSelected = false;
        }

        public void SetJpegSelected()
        {
            JpegSelected = true;
            WebpSelected = false;
            PngSelected = false;
            AvifSelected = false;
        }

        public void SetAvifSelected()
        {
            AvifSelected = true;
            JpegSelected = false;
            WebpSelected = false;
            PngSelected = false;
        }

        public void SetModeScaleSelected()
        {
            ModeScaleSelected = true;
            ModeWidthSelected = false;
            ModeHeightSelected = false;
            ModeFitToDisplaySelected = false;
        }

        public void SetModeWidthSelected()
        {
            ModeWidthSelected = true;
            ModeScaleSelected = false;
            ModeHeightSelected = false;
            ModeFitToDisplaySelected = false;
        }

        public void SetModeHeightSelected()
        {
            ModeHeightSelected = true;
            ModeScaleSelected = false;
            ModeWidthSelected = false;
            ModeFitToDisplaySelected = false;
        }

        public void SetModeFitToDisplaySelected()
        {
            ModeFitToDisplaySelected = true;
            ModeHeightSelected = false;
            ModeWidthSelected = false;
            ModeScaleSelected = false;
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

            if (ModeHeightSelected && ResizeHeightAfterUpscale == 0)
            {
                valid = false;
                validationText.Add("Output Height is invalid. Enter a height larger than 0.");
            }

            if (ModeWidthSelected && ResizeWidthAfterUpscale == 0)
            {
                valid = false;
                validationText.Add("Output Width is invalid. Enter a width larger than 0.");
            }

            if (ModeFitToDisplaySelected && (DisplayDeviceWidth == 0 || DisplayDeviceHeight == 0))
            {
                valid = false;
                validationText.Add("Tablet Device or Display is invalid. Please make a selection from the list of options.");
            }

            Valid = valid;

            if (Vm != null)
            {
                // TODO
                Vm.CheckInputs();
                if (Vm?.ProgressTotalFiles == 0)
                {
                    Valid = false;
                    validationText.Add($"{Vm?.InputStatusText} selected for upscaling. At least one file must be selected.");
                }
                Vm.ValidationText = string.Join("\n", validationText);
            }
        }
    }

    [DataContract]
    public class UpscaleChain : ReactiveObject
    {
        IPythonService _pythonService;

        public UpscaleChain(IPythonService? pythonService = null)
        {
            _pythonService = pythonService ?? Locator.Current.GetService<IPythonService>()!;

            this.WhenAnyValue(x => x.Vm).Subscribe(x =>
            {
                sub?.Dispose();
                sub = Vm.WhenAnyValue(
                    x => x.IsExtractingBackend
                    ).Subscribe(x =>
                    {
                        this.RaisePropertyChanged(nameof(AllModels));
                        this.RaisePropertyChanged(nameof(ModelFilePath));
                    });
            });

            this.RaisePropertyChanged(nameof(AllModels));
            this.RaisePropertyChanged(nameof(ModelFilePath));
        }

        private IDisposable? sub;

        private MainWindowViewModel? _vm;
        public MainWindowViewModel? Vm
        {
            get => _vm;
            set => this.RaiseAndSetIfChanged(ref _vm, value);
        }

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

        public AvaloniaList<string> AllModels => _pythonService.AllModels;

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
    }

    // TODO refactor into separate file
    public class ReaderDevice
    {
        public string Name { get; set; } = default!;
        public string Brand { get; set; } = default!;
        public string Year { get; set; } = default!;
        public int Width { get; set; } = default!;
        public int Height { get; set; } = default!;

        public override string ToString()
        {
            List<string> parts = [];

            if (!string.IsNullOrWhiteSpace(Brand))
            {
                parts.Add(Brand);
            }

            if (!string.IsNullOrWhiteSpace(Name))
            {
                parts.Add(Name);
            }

            if (!string.IsNullOrWhiteSpace(Year))
            {
                parts.Add($"({Year})");
            }

            return string.Join(" ", parts);
        }
    }
}
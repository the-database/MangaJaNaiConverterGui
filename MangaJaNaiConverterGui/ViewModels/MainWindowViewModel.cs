using Avalonia.Collections;
using Avalonia.Threading;
using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.Services;
using Newtonsoft.Json;
using ReactiveUI;
using ReactiveUI.Primitives;
using Splat;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.Serialization;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
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

        // Auto update state: the latest release's version, the version staged by the background
        // download (installed when the app closes), and the download in flight.
        private string? _availableUpdateVersion;
        private string? _readyUpdateVersion;
        private CancellationTokenSource? _updateDownloadCts;
        private string _updateDownloadProgress = "";
        private string? _installError;
        private string? _downloadError;
        private bool _updateChecksStarted;
        private static readonly TimeSpan UpdateCheckInterval = TimeSpan.FromHours(6);

        private readonly IBackendService _backendService;
        private readonly IUpdateManagerService _updateManagerService;
        private readonly ISuspensionDriverService _suspensionDriverService;

        public MainWindowViewModel(IBackendService? backendService = null, IUpdateManagerService? updateManagerService = null, ISuspensionDriverService? suspensionDriverService = null)
        {
            _backendService = backendService ?? Locator.Current.GetService<IBackendService>()!;
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

            this.WhenAnyValue(x => x.ShowAppSettings).Subscribe(async x =>
            {
                if (x)
                {
                    await RefreshComponents();
                }
            });
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
        "ModelFilePath": "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx",
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
        "ModelFilePath": "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_1200p_V1_ESRGAN_70k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_1200p_V1_ESRGAN_70k.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_1300p_V1_ESRGAN_75k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_1300p_V1_ESRGAN_75k.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_1400p_V1_ESRGAN_70k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_1400p_V1_ESRGAN_105k.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_1500p_V1_ESRGAN_90k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_1500p_V1_ESRGAN_105k.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_1600p_V1_ESRGAN_90k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_1600p_V1_ESRGAN_70k.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_1920p_V1_ESRGAN_70k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_1920p_V1_ESRGAN_105k.onnx",
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
        "ModelFilePath": "2x_MangaJaNai_2048p_V1_ESRGAN_95k.onnx",
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
        "ModelFilePath": "4x_MangaJaNai_2048p_V1_ESRGAN_70k.onnx",
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
        // The user's models folder; the backend also looks in the bundled one.
        [DataMember]
        public string ModelsDirectory => _backendService.UserModelsDirectory;

        private bool _showCheckUpdateButton = true;
        public bool ShowCheckUpdateButton
        {
            get => _showCheckUpdateButton;
            set => this.RaiseAndSetIfChanged(ref _showCheckUpdateButton, value);
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

        // Each tile size is its own TensorRT engine (built once per model on first use), so only a few.
        private string[] _tileSizes = [
            "Auto (Estimate)",
            "256",
            "384",
            "512",
            "768",
            "1024"];

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

        private AvaloniaDictionary<string, ReaderDevice> _displayDeviceMap = [];
        [DataMember]
        public AvaloniaDictionary<string, ReaderDevice> DisplayDeviceMap
        {
            get => _displayDeviceMap;
            set => this.RaiseAndSetIfChanged(ref _displayDeviceMap, value);
        }

        // The console's height, dragged with the handle above it.
        private double _consoleHeight = 300;
        [DataMember]
        public double ConsoleHeight
        {
            get => _consoleHeight;
            set => this.RaiseAndSetIfChanged(ref _consoleHeight, value);
        }

        // The workflow sidebar's width, dragged with its splitter (app-wide, not per workflow).
        private double _sidebarWidth = 220;
        [DataMember]
        public double SidebarWidth
        {
            get => _sidebarWidth;
            set => this.RaiseAndSetIfChanged(ref _sidebarWidth, value);
        }

        private bool _autoUpdate = true;
        [DataMember]
        public bool AutoUpdateEnabled
        {
            get => _autoUpdate;
            set => this.RaiseAndSetIfChanged(ref _autoUpdate, value);
        }

        // Which inference backend upscales: Auto (TensorRT when it can run, else DirectML), TensorRT or
        // DirectML.
        private string _upscaleBackend = "Auto";
        [DataMember]
        public string UpscaleBackend
        {
            get => _upscaleBackend;
            set
            {
                this.RaiseAndSetIfChanged(ref _upscaleBackend, value);
                this.RaisePropertyChanged(nameof(AutoBackendSelected));
                this.RaisePropertyChanged(nameof(TensorRtBackendSelected));
                this.RaisePropertyChanged(nameof(DirectMlBackendSelected));
            }
        }

        public bool AutoBackendSelected => UpscaleBackend != "TensorRT" && UpscaleBackend != "DirectML";
        public bool TensorRtBackendSelected => UpscaleBackend == "TensorRT";
        public bool DirectMlBackendSelected => UpscaleBackend == "DirectML";

        public async Task SetBackendAuto() => await SetBackend("Auto");
        public async Task SetBackendTensorRt() => await SetBackend("TensorRT");
        public async Task SetBackendDirectMl() => await SetBackend("DirectML");

        private async Task SetBackend(string backend)
        {
            UpscaleBackend = backend;
            await RefreshDevices();
        }

        // The backend the device list belongs to (Auto resolves to the one it found devices on).
        private string _deviceBackend = "tensorrt";
        private string[] _deviceIds = [];
        private string[] _deviceNames = [];

        private int _selectedDeviceIndex;
        [DataMember]
        public int SelectedDeviceIndex
        {
            get => _selectedDeviceIndex;
            set
            {
                this.RaiseAndSetIfChanged(ref _selectedDeviceIndex, value);
                if (value >= 0 && value < DeviceList.Length)
                {
                    SelectedDeviceName = _deviceNames[value];
                    SelectedDeviceId = _deviceIds[value];
                }
            }
        }

        // The selected device's id: the card's PCI address (animejanai-inference 0.18.2+), the same on
        // both backends and across reboots. It changes only when the card moves slot, and older
        // releases' DirectML ids changed every boot: a saved id that's gone falls back to
        // SelectedDeviceIndex and SelectedDeviceName.
        private string? _selectedDeviceId;
        [DataMember]
        public string? SelectedDeviceId
        {
            get => _selectedDeviceId;
            set => this.RaiseAndSetIfChanged(ref _selectedDeviceId, value);
        }

        // The version this state was last used with, so an update can refresh components once.
        private string? _lastRunVersion;
        [DataMember]
        public string? LastRunVersion
        {
            get => _lastRunVersion;
            set => this.RaiseAndSetIfChanged(ref _lastRunVersion, value);
        }

        // The selected device's name (as the backend lists it, without the label's id suffix).
        private string? _selectedDeviceName;
        [DataMember]
        public string? SelectedDeviceName
        {
            get => _selectedDeviceName;
            set => this.RaiseAndSetIfChanged(ref _selectedDeviceName, value);
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
                this.RaisePropertyChanged(nameof(AllowComponentChanges));
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

        public string LeftStatus =>
            !string.IsNullOrEmpty(EngineStatusText) ? EngineStatusText :
            !CurrentWorkflow.Valid ? ValidationText.Replace("\n", " ") : $"{InputStatusText} selected for upscaling.";

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

        public void CloseAppSettings()
        {
            RequestShowAppSettings = false;
        }

        // ---- component manager (App Settings) ------------------------------------------------

        private bool _componentsBusy = false;
        public bool ComponentsBusy
        {
            get => _componentsBusy;
            set
            {
                this.RaiseAndSetIfChanged(ref _componentsBusy, value);
                this.RaisePropertyChanged(nameof(AllowComponentChanges));
            }
        }

        public bool AllowComponentChanges => !ComponentsBusy && !Upscaling;

        private AvaloniaList<ComponentItem> _components = [];
        public AvaloniaList<ComponentItem> Components
        {
            get => _components;
            set => this.RaiseAndSetIfChanged(ref _components, value);
        }

        private string _gpuStatusText = string.Empty;
        public string GpuStatusText
        {
            get => _gpuStatusText;
            set => this.RaiseAndSetIfChanged(ref _gpuStatusText, value);
        }

        private string _componentStatusText = string.Empty;
        public string ComponentStatusText
        {
            get => _componentStatusText;
            set => this.RaiseAndSetIfChanged(ref _componentStatusText, value);
        }

        public async Task RefreshComponents()
        {
            try
            {
                var json = await _updateManagerService.GetComponentsJsonAsync();
                if (string.IsNullOrWhiteSpace(json))
                {
                    return;
                }
                var root = Newtonsoft.Json.Linq.JObject.Parse(json);
                var gpu = root["gpu"];
                bool nvidia = (bool?)gpu?["nvidia"] ?? false;
                var gpuName = (string?)gpu?["name"] ?? "";
                var sm = (string?)gpu?["sm"] ?? "";
                GpuStatusText = nvidia
                    ? $"{gpuName} ({sm})"
                    : "No NVIDIA GPU detected — using the built-in DirectML engine.";

                var items = new AvaloniaList<ComponentItem>();
                foreach (var p in (Newtonsoft.Json.Linq.JArray?)root["packs"] ?? [])
                {
                    var name = (string?)p["name"] ?? "";
                    var installed = (bool?)p["installed"] ?? false;
                    var recommended = (bool?)p["recommended"] ?? false;
                    // Only surface packs relevant to this machine: what it uses (recommended) or
                    // already has (installed, so it can be removed). Kernel packs for other GPU
                    // generations stay hidden; the updater CLI still lists them all.
                    if (!installed && !recommended)
                    {
                        continue;
                    }
                    items.Add(new ComponentItem(name, (long?)p["bytes"] ?? 0, installed, recommended));
                }
                Components = items;
            }
            catch (Exception ex)
            {
                ComponentStatusText = $"Could not read components: {ex.Message}";
            }
        }

        public async Task InstallComponent(ComponentItem item)
        {
            await RunComponentCommand(item, $"--install {item.Name}", $"Installing {item.Title}...");
        }

        public async Task RemoveComponent(ComponentItem item)
        {
            await RunComponentCommand(item, $"--remove {item.Name}", $"Removing {item.Title}...");
        }

        private async Task RunComponentCommand(ComponentItem item, string arguments, string status)
        {
            if (item is null || ComponentsBusy)
            {
                return;
            }
            ComponentsBusy = true;
            ComponentStatusText = status;
            try
            {
                await _updateManagerService.RunUpdaterStreamingAsync(arguments, line => ComponentStatusText = line);
            }
            catch (Exception ex) { ComponentStatusText = ex.Message; }
            ComponentsBusy = false;
            _backendService.RefreshModels();
            await RefreshComponents();
            await RefreshDevices();
        }




        public async Task RunUpscale()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;

            var task = Task.Run(async () =>
            {
                // The backend reads its settings from the saved state file.
                _suspensionDriverService.SuspensionDriver.SaveState(this);
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

                if (!_backendService.IsBackendInstalled)
                {
                    ConsoleQueueEnqueue($"The upscaling engine is missing ({_backendService.BackendExePath}). Install it from App Settings > Components.");
                    return;
                }
                Directory.CreateDirectory(_backendService.EngineCacheDirectory);
                var args = new List<string>
                {
                    "--settings", _backendService.AppStatePath,
                    "--aji-dir", _backendService.InferenceDirectory,
                    "--backend", _deviceIds.Length > 0 ? _deviceBackend : "auto",
                    "--cache-dir", _backendService.EngineCacheDirectory,
                    "--models-dir", _backendService.UserModelsDirectory,
                    "--models-dir", _backendService.BundledModelsDirectory,
                };
                if (_deviceIds.Length > 0 && !string.IsNullOrEmpty(SelectedDeviceId))
                {
                    args.AddRange(["--device", SelectedDeviceId]);
                }
                ConsoleQueueEnqueue($"Upscaling with: {_backendService.BackendExePath} {string.Join(' ', args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a))}");
                await RunCommand(args);

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
                                                        OutputDirectoryFor(inputImagePath),
                                                        CurrentWorkflow.OutputFilename.Replace("%filename%", Path.GetFileNameWithoutExtension(inputImagePath))) + $".{CurrentWorkflow.ImageFormat}";
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
                                                            OutputDirectoryFor(inputArchivePath),
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

        // A folder upscale mirrors the input's subfolders in the output folder.
        private string OutputDirectoryFor(string inputPath)
        {
            var relative = Path.GetRelativePath(Path.GetFullPath(CurrentWorkflow.InputFolderPath), Path.GetDirectoryName(Path.GetFullPath(inputPath)) ?? "");
            return Path.GetFullPath(Path.Join(CurrentWorkflow.OutputFolderPath, relative == "." ? "" : relative));
        }

        public void AddChain()
        {
            CurrentWorkflow?.Chains.Add(new UpscaleChain
            {
                Vm = this,
                ChainSectionExpanded = true,
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



        public async Task RunCommand(IEnumerable<string> arguments)
        {
            using (var process = new Process())
            {
                _runningProcess = process;
                process.StartInfo.FileName = _backendService.BackendExePath;
                foreach (var a in arguments)
                {
                    process.StartInfo.ArgumentList.Add(a);
                }
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.WorkingDirectory = _backendService.InferenceDirectory;
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                process.StartInfo.StandardErrorEncoding = Encoding.UTF8;

                Directory.CreateDirectory(_backendService.LogsDirectory);
                using (var outputFile = new StreamWriter(Path.Combine(_backendService.LogsDirectory, "upscale.log"), append: false))
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
                                // A first-use TensorRT engine build takes minutes: say so in the status bar.
                                if (e.Data.StartsWith("ENGINE_BUILD "))
                                {
                                    EngineStatusText = $"Building the TensorRT engine for {e.Data["ENGINE_BUILD ".Length..].Split(' ')[0]} (a few minutes, once per model)...";
                                }
                                else if (e.Data.StartsWith("ENGINE_READY "))
                                {
                                    EngineStatusText = string.Empty;
                                }
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
                    EngineStatusText = string.Empty;
                }

            }
        }

        private string _engineStatusText = string.Empty;
        public string EngineStatusText
        {
            get => _engineStatusText;
            set
            {
                this.RaiseAndSetIfChanged(ref _engineStatusText, value);
                this.RaisePropertyChanged(nameof(LeftStatus));
            }
        }

        // Run the backend for a short query (e.g. --list-devices) and return its stdout.
        private async Task<string> RunBackendQuery(params string[] arguments)
        {
            if (!_backendService.IsBackendInstalled)
            {
                return string.Empty;
            }
            using var process = new Process();
            process.StartInfo.FileName = _backendService.BackendExePath;
            foreach (var a in arguments)
            {
                process.StartInfo.ArgumentList.Add(a);
            }
            process.StartInfo.RedirectStandardOutput = true;
            process.StartInfo.RedirectStandardError = true;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.CreateNoWindow = true;
            process.StartInfo.WorkingDirectory = _backendService.InferenceDirectory;
            process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
            process.Start();
            var stdout = process.StandardOutput.ReadToEndAsync();
            _ = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            return await stdout;
        }

        private async Task<List<BackendDevice>> ListDevices(string backend)
        {
            try
            {
                var json = await RunBackendQuery("--list-devices", "--backend", backend, "--aji-dir", _backendService.InferenceDirectory);
                return string.IsNullOrWhiteSpace(json) ? [] : JsonConvert.DeserializeObject<List<BackendDevice>>(json) ?? [];
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
                return [];
            }
        }

        // Fill the device list for the chosen backend (Auto: TensorRT's devices if it can run here,
        // else DirectML's), keeping the user's device while it's still there.
        public async Task RefreshDevices()
        {
            var backends = UpscaleBackend switch
            {
                "TensorRT" => new[] { "tensorrt" },
                "DirectML" => new[] { "directml" },
                _ => new[] { "tensorrt", "directml" },
            };
            List<BackendDevice> devices = [];
            var backend = backends[0];
            foreach (var b in backends)
            {
                devices = await ListDevices(b);
                backend = b;
                if (devices.Count > 0)
                {
                    break;
                }
            }

            var selected = BackendDevice.Find(devices, SelectedDeviceId, SelectedDeviceName, SelectedDeviceIndex);
            _deviceBackend = backend;
            _deviceIds = devices.Select(d => d.Id).ToArray();
            _deviceNames = devices.Select(d => d.Name).ToArray();
            // Replacing the items clears the ComboBox's selection; select after, as a real change,
            // so the ComboBox picks it up even when the index is the same as before.
            _selectedDeviceIndex = -1;
            DeviceList = BackendDevice.Labels(devices);
            if (devices.Count > 0)
            {
                SelectedDeviceIndex = selected;
            }
            DeviceStatusText = devices.Count == 0
                ? "No compatible GPU found. MangaJaNaiConverterGui needs an NVIDIA GPU (TensorRT) or a DirectX 12 GPU (DirectML)."
                : $"Using {(backend == "tensorrt" ? "TensorRT" : "DirectML")}.";
        }

        private string _deviceStatusText = string.Empty;
        public string DeviceStatusText
        {
            get => _deviceStatusText;
            set => this.RaiseAndSetIfChanged(ref _deviceStatusText, value);
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
                foreach (var chain in workflow.Chains)
                {
                    chain.Migrate();
                }
                ReplaceCurrentWorkflow(workflow);
            }
        }

        // Wire up the new workflow and detach the replaced one so its subscriptions on this view model are disposed.
        private void ReplaceCurrentWorkflow(UpscaleWorkflow workflow)
        {
            workflow.Vm = this;
            foreach (var chain in workflow.Chains)
            {
                chain.Vm = this;
            }

            var old = CurrentWorkflow;
            CurrentWorkflow = workflow;
            if (old != null)
            {
                old.Vm = null;
                foreach (var chain in old.Chains)
                {
                    chain.Vm = null;
                }
            }
            workflow.Validate();
        }

        public void WriteCurrentWorkflowToFile(string fullPath)
        {
            var lines = JsonConvert.SerializeObject(CurrentWorkflow, NewtonsoftJsonSuspensionDriver.Settings);
            File.WriteAllText(fullPath, lines);
        }

        // On launch: an installed copy without its upscaling engine or models (a first run whose
        // component download failed, or components removed), or just updated to a new version, runs
        // the updater's --auto to install what this GPU needs. Then the models and devices are read.
        public async Task CheckAndExtractBackend()
        {
            var version = _updateManagerService.AppVersion;
            var updated = !string.IsNullOrEmpty(LastRunVersion) && LastRunVersion != version;
            if (_updateManagerService.IsInstalled && (!_backendService.IsBackendInstalled || !_backendService.AreModelsInstalled || updated))
            {
                IsExtractingBackend = true;
                BackendSetupMainStatus = "Setting up the upscaling components for this GPU...";
                try
                {
                    await Task.Run(() => _updateManagerService.RunUpdaterStreamingAsync("--auto", BackendSetupSubStatusQueueEnqueue));
                }
                catch (Exception ex)
                {
                    BackendSetupSubStatusQueueEnqueue(ex.Message);
                }
            }
            LastRunVersion = version;

            _backendService.RefreshModels();
            IsExtractingBackend = false;
            await RefreshDevices();
            CurrentWorkflow?.Validate();
        }

        public void CheckAndDoBackup()
        {
            Task.Run(() =>
            {
                try
                {
                    if (!File.Exists(_backendService.AppStatePath))
                        return;

                    var files = Directory.EnumerateFiles(_backendService.AppStateFolder)
                        .Where(f =>
                        {
                            var name = Path.GetFileName(f);
                            return name.StartsWith("autobackup_") &&
                                   name.EndsWith(_backendService.AppStateFilename);
                        })
                        .OrderByDescending(f => f)
                        .ToList();

                    var latestBackup = files.FirstOrDefault();
                    if (latestBackup is not null &&
                        FilesAreEqual(_backendService.AppStatePath, latestBackup))
                    {
                        return;
                    }

                    var backupName =
                        $"autobackup_{DateTime.Now:yyyyMMdd-HHmmss}_{_backendService.AppStateFilename}";
                    var backupPath = Path.Combine(_backendService.AppStateFolder, backupName);

                    File.Copy(_backendService.AppStatePath, backupPath);

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
                    ReplaceCurrentWorkflow(new UpscaleWorkflow
                    {
                        WorkflowIndex = workflowIndex,
                        WorkflowName = workflowName,
                        Chains = workflow.Chains
                    });
                }
            }
        }

        // Avalonia 12 compiled bindings only convert zero/one-parameter methods to delegates.
        public Func<string?, CancellationToken, Task<IEnumerable<object>>> DevicesPopulator => PopulateDevicesAsync;

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

        // Called once for the app's state (not every view model instance): check now, then
        // periodically, downloading in the background while Auto Update is on.
        public void StartUpdateChecks()
        {
            if (_updateChecksStarted || !_updateManagerService.IsInstalled)
            {
                return;
            }
            _updateChecksStarted = true;
            if (_updateManagerService.TakeLastUpdateError() is string error)
            {
                _installError = $"The last update failed to install: {error}";
            }
            bool autoUpdate = AutoUpdateEnabled;
            this.WhenAnyValue(x => x.AutoUpdateEnabled).Subscribe(on =>
            {
                if (on == autoUpdate)
                {
                    return; // the current value, delivered on subscribe
                }
                autoUpdate = on;
                if (on)
                {
                    StartUpdateDownload();
                }
                else
                {
                    CancelUpdateDownload();
                }
                UpdateStatus();
            });
            DispatcherTimer.Run(() =>
            {
                _ = CheckForUpdates();
                return true;
            }, UpdateCheckInterval);
            _ = CheckForUpdates();
        }

        public async Task CheckForUpdates()
        {
            try
            {
                if (_updateManagerService.IsInstalled)
                {
                    _availableUpdateVersion = await _updateManagerService.CheckForUpdateAsync().ConfigureAwait(true);
                    if (_readyUpdateVersion != _availableUpdateVersion)
                    {
                        _readyUpdateVersion = null; // a newer release replaced the staged one
                    }
                    UpdateStatus();
                    StartUpdateDownload();
                }
            }
            catch (Exception ex)
            {
                UpdateStatusText = $"Check for update failed: {ex.Message}";
            }
        }

        private void StartUpdateDownload()
        {
            if (!AutoUpdateEnabled || _availableUpdateVersion is null || _readyUpdateVersion == _availableUpdateVersion ||
                _updateDownloadCts != null)
            {
                return;
            }
            var cts = _updateDownloadCts = new CancellationTokenSource();
            _updateDownloadProgress = "";
            UpdateStatus();
            _ = Task.Run(async () =>
            {
                string? ready = null;
                string? failure = null;
                string lastLine = "";
                try
                {
                    ready = await _updateManagerService.DownloadUpdateAsync(line =>
                    {
                        lastLine = line;
                        var percent = Regex.Match(line, @"(\d+)%$");
                        var progress = percent.Success ? $"{percent.Groups[1].Value}%"
                            : line.StartsWith("Extracting") ? "unpacking" : null;
                        if (progress != null)
                        {
                            Dispatcher.UIThread.Post(() =>
                            {
                                _updateDownloadProgress = progress;
                                UpdateStatus();
                            });
                        }
                    }, cts.Token);
                    if (ready is null)
                    {
                        failure = lastLine;
                    }
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    failure = ex.Message;
                }
                Dispatcher.UIThread.Post(() =>
                {
                    if (_updateDownloadCts == cts)
                    {
                        _updateDownloadCts = null;
                    }
                    cts.Dispose();
                    if (ready != null)
                    {
                        _readyUpdateVersion = ready;
                        _downloadError = null;
                    }
                    else if (failure != null)
                    {
                        _downloadError = $"Downloading the update failed: {failure}";
                    }
                    UpdateStatus();
                });
            });
        }

        private void CancelUpdateDownload()
        {
            _updateDownloadCts?.Cancel();
            _updateDownloadCts = null;
        }

        // The bundled updater downloads (unless already staged) + applies in one step, then relaunches.
        // Save the settings, launch it and exit so it can replace files.
        public void ApplyUpdate()
        {
            if (Upscaling)
            {
                return;
            }
            CancelUpdateDownload();
            ShowApplyButton = false;
            UpdateStatusText = "Updating and restarting...";
            // Environment.Exit skips the save a normal exit does.
            _suspensionDriverService.SuspensionDriver.SaveState(this);
            _updateManagerService.ApplyUpdateAndRestart();
            Environment.Exit(0);
        }

        // On close: install an update the background download has staged.
        public void OnAppExit()
        {
            CancelUpdateDownload();
            if (AutoUpdateEnabled && _readyUpdateVersion != null)
            {
                _updateManagerService.ApplyUpdateOnExit();
            }
        }

        private void UpdateStatus()
        {
            string status;
            if (string.IsNullOrEmpty(_availableUpdateVersion))
            {
                status = "No updates found";
            }
            else if (_updateDownloadCts != null)
            {
                status = string.IsNullOrEmpty(_updateDownloadProgress)
                    ? $"Downloading update {_availableUpdateVersion}..."
                    : $"Downloading update {_availableUpdateVersion} ({_updateDownloadProgress})...";
            }
            else if (_readyUpdateVersion == _availableUpdateVersion && AutoUpdateEnabled)
            {
                status = $"Version {_availableUpdateVersion} is ready and will install when you close MangaJaNaiConverterGui.";
            }
            else
            {
                status = $"Update available: {_availableUpdateVersion}";
            }
            UpdateStatusText = string.Join("\n", new[] { _installError, _downloadError, status }.Where(x => x != null));
            ShowApplyButton = !string.IsNullOrEmpty(_availableUpdateVersion);
            ShowCheckUpdateButton = !ShowApplyButton;
        }

        public async void OpenModelsDirectory()
        {
            await Task.Run(() =>
            {
                Directory.CreateDirectory(_backendService.UserModelsDirectory);
                Process.Start("explorer.exe", _backendService.UserModelsDirectory);
            });
        }
    }

    // A downloadable component pack, surfaced in the App Settings component manager.
    public class ComponentItem(string name, long bytes, bool installed, bool recommended)
    {
        public string Name { get; } = name;
        public long Bytes { get; } = bytes;
        public bool Installed { get; } = installed;
        public bool Recommended { get; } = recommended;

        // User-facing label for the raw pack id. Only the packs relevant to this machine reach the
        // UI, so the GPU family shown is the user's own.
        public string Title => Name switch
        {
            "trt-runtime" => "TensorRT runtime",
            "trt-ptx" => "TensorRT kernels: other NVIDIA GPUs",
            "models-mangajanai" => "MangaJaNai models (grayscale manga)",
            "models-illustrationjanai" => "IllustrationJaNai models (color)",
            _ when Name.StartsWith("trt-sm") => $"TensorRT kernels: {SmFamily(Name[6..])}",
            _ => Name,
        };

        public string Description => Name switch
        {
            "trt-runtime" => "The fastest upscaling engine, for NVIDIA GPUs. Without it, upscaling uses the slower DirectML engine.",
            "trt-ptx" => "Fallback kernels for NVIDIA GPUs without a dedicated kernel pack. The first engine build is slower.",
            "models-mangajanai" => "The default models for grayscale manga pages, one per source resolution.",
            "models-illustrationjanai" => "The default models for color pages and illustrations.",
            _ when Name.StartsWith("trt-sm") => "Engine-builder kernels matched to your GPU generation.",
            _ => "",
        };

        // CUDA compute capability (sm) -> consumer GPU family.
        private static string SmFamily(string sm) => sm switch
        {
            "75" => "GeForce RTX 20 series (Turing)",
            "80" or "86" => "GeForce RTX 30 series (Ampere)",
            "89" => "GeForce RTX 40 series (Ada)",
            "90" => "Hopper",
            "100" or "120" => "GeForce RTX 50 series (Blackwell)",
            _ => $"sm{sm}",
        };

        public string SizeText => $"{Bytes / 1048576} MB";
        public string StateText => Installed ? "installed" : Recommended ? "recommended" : "available";
        public bool CanInstall => !Installed;
        public bool CanRemove => Installed;
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

            g1.Subscribe(x => Validate());
            g2.Subscribe(x => Validate());
            g3.Subscribe(x => Validate());

            this.WhenAnyValue(x => x.Vm).Subscribe(vm =>
            {
                sub?.Dispose();
                sub = null;
                // Vm is null until App wires it up after loading (and while deserializing);
                // ReactiveUI 26's generated WhenAnyValue throws on a null source.
                if (vm is null)
                {
                    return;
                }
                sub = vm.WhenAnyValue(
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

            // The collapsed sections' summaries follow every setting.
            PropertyChanged += (_, e) =>
            {
                if (e.PropertyName is not null && !e.PropertyName.EndsWith("Summary", StringComparison.Ordinal))
                {
                    RaiseSummaries();
                }
            };
        }

        // ---- collapsible sections ----------------------------------------------------------

        private bool _inputSectionExpanded = true;
        [DataMember]
        public bool InputSectionExpanded
        {
            get => _inputSectionExpanded;
            set => this.RaiseAndSetIfChanged(ref _inputSectionExpanded, value);
        }

        private bool _upscaleSectionExpanded = true;
        [DataMember]
        public bool UpscaleSectionExpanded
        {
            get => _upscaleSectionExpanded;
            set => this.RaiseAndSetIfChanged(ref _upscaleSectionExpanded, value);
        }

        private bool _outputSectionExpanded = true;
        [DataMember]
        public bool OutputSectionExpanded
        {
            get => _outputSectionExpanded;
            set => this.RaiseAndSetIfChanged(ref _outputSectionExpanded, value);
        }

        // The Model Chains section is open when ShowAdvancedSettings is on (the saved setting that
        // used to show those settings), so existing workflows keep how they looked.

        public string InputSummary
        {
            get
            {
                if (SelectedTabIndex == 0)
                {
                    return string.IsNullOrWhiteSpace(InputFilePath) ? "Single file · none selected" : $"Single file · {Path.GetFileName(InputFilePath)}";
                }
                var kinds = (UpscaleArchives, UpscaleImages) switch
                {
                    (true, true) => "archives + images",
                    (true, false) => "archives",
                    (false, true) => "images",
                    _ => "nothing selected",
                };
                return string.IsNullOrWhiteSpace(InputFolderPath) ? $"Batch folder · none selected" : $"Batch folder · {InputFolderPath} · {kinds}";
            }
        }

        public string UpscaleSummary
        {
            get
            {
                if (ModeWidthSelected)
                {
                    return $"Width {ResizeWidthAfterUpscale} px";
                }
                if (ModeHeightSelected)
                {
                    return $"Height {ResizeHeightAfterUpscale} px";
                }
                if (ModeFitToDisplaySelected)
                {
                    var orientation = DisplayPortraitSelected ? "portrait" : "landscape";
                    return string.IsNullOrWhiteSpace(DisplayDevice) ? "Fit to display" : $"Fit to {DisplayDevice} · {orientation}";
                }
                return $"Scale {UpscaleScaleFactor}x";
            }
        }

        public string ChainsSummary
        {
            get
            {
                var count = Chains?.Count ?? 0;
                return $"{count} {(count == 1 ? "chain" : "chains")} · grayscale threshold {GrayscaleDetectionThreshold}";
            }
        }

        public string OutputSummary
        {
            get
            {
                var format = WebpSelected ? "WebP" : PngSelected ? "PNG" : AvifSelected ? "AVIF" : "JPEG";
                var quality = ShowLossyCompressionQuality ? $" q{LossyCompressionQuality}" : ShowUseLosslessCompression ? " lossless" : "";
                return string.IsNullOrWhiteSpace(OutputFolderPath) ? $"{format}{quality}" : $"{format}{quality} → {OutputFolderPath}";
            }
        }

        public void ExpandAllSections() => SetAllSectionsExpanded(true);

        public void CollapseAllSections() => SetAllSectionsExpanded(false);

        private void SetAllSectionsExpanded(bool expanded)
        {
            InputSectionExpanded = expanded;
            UpscaleSectionExpanded = expanded;
            ShowAdvancedSettings = expanded;
            OutputSectionExpanded = expanded;
            foreach (var chain in Chains)
            {
                chain.ChainSectionExpanded = expanded;
            }
        }

        private void RaiseSummaries()
        {
            this.RaisePropertyChanged(nameof(InputSummary));
            this.RaisePropertyChanged(nameof(UpscaleSummary));
            this.RaisePropertyChanged(nameof(ChainsSummary));
            this.RaisePropertyChanged(nameof(OutputSummary));
        }

        private void ChainsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) => this.RaisePropertyChanged(nameof(ChainsSummary));

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
            set
            {
                if (_chains != null)
                {
                    _chains.CollectionChanged -= ChainsCollectionChanged;
                }
                this.RaiseAndSetIfChanged(ref _chains, value);
                if (_chains != null)
                {
                    _chains.CollectionChanged += ChainsCollectionChanged;
                }
            }
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

            foreach (var chain in Chains ?? [])
            {
                if (!chain.HasOnnxModel)
                {
                    valid = false;
                    validationText.Add($"Chain {chain.ChainNumber}'s model {chain.ModelFilePath} is a PyTorch model; MangaJaNaiConverterGui 2.0 runs ONNX models. Convert it to ONNX: {UpscaleChain.ConvertModelsUrl}");
                }
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
        IBackendService _backendService;

        public UpscaleChain(IBackendService? backendService = null)
        {
            _backendService = backendService ?? Locator.Current.GetService<IBackendService>()!;

            this.WhenAnyValue(x => x.Vm).Subscribe(vm =>
            {
                sub?.Dispose();
                sub = null;
                if (vm is null)
                {
                    return;
                }
                sub = vm.WhenAnyValue(
                    x => x.IsExtractingBackend
                    ).Subscribe(x =>
                    {
                        this.RaisePropertyChanged(nameof(AllModels));
                        this.RaisePropertyChanged(nameof(ModelFilePath));
                    });
            });

            this.RaisePropertyChanged(nameof(AllModels));
            this.RaisePropertyChanged(nameof(ModelFilePath));

            // The collapsed chains' summaries follow every setting.
            PropertyChanged += (_, e) =>
            {
                if (e.PropertyName is { } name && !name.EndsWith("Summary", StringComparison.Ordinal) && !name.EndsWith("Expanded", StringComparison.Ordinal))
                {
                    this.RaisePropertyChanged(nameof(ChainSummary));
                    this.RaisePropertyChanged(nameof(ActivationSummary));
                    this.RaisePropertyChanged(nameof(UpscaleSummary));
                }
            };
        }

        private IDisposable? sub;

        private MainWindowViewModel? _vm;
        public MainWindowViewModel? Vm
        {
            get => _vm;
            set => this.RaiseAndSetIfChanged(ref _vm, value);
        }

        // ---- collapsible sections ----------------------------------------------------------
        // Saved with the chain (the backend ignores these keys). The default workflow has 16 chains,
        // so a chain starts collapsed, showing its summary; its inner sections start open.

        private bool _chainSectionExpanded = false;
        [DataMember]
        public bool ChainSectionExpanded
        {
            get => _chainSectionExpanded;
            set => this.RaiseAndSetIfChanged(ref _chainSectionExpanded, value);
        }

        private bool _activationSectionExpanded = true;
        [DataMember]
        public bool ActivationSectionExpanded
        {
            get => _activationSectionExpanded;
            set => this.RaiseAndSetIfChanged(ref _activationSectionExpanded, value);
        }

        private bool _upscaleSectionExpanded = true;
        [DataMember]
        public bool UpscaleSectionExpanded
        {
            get => _upscaleSectionExpanded;
            set => this.RaiseAndSetIfChanged(ref _upscaleSectionExpanded, value);
        }

        // ---- header summaries ------------------------------------------------------------------

        private static string RangeText(string? min, string? max, string any, string unit = "")
        {
            bool noMin = string.IsNullOrEmpty(min) || min == "0" || min == "0x0";
            bool noMax = string.IsNullOrEmpty(max) || max == "0" || max == "0x0";
            return (noMin, noMax) switch
            {
                (true, true) => any,
                (false, true) => $"{min}{unit}+",
                (true, false) => $"up to {max}{unit}",
                _ => $"{min}{unit} - {max}{unit}",
            };
        }

        // "0x1250" -> "1250p", "800x0" -> "800px wide"; a 0 dimension means any.
        private static string? ResolutionText(string? resolution)
        {
            var parts = resolution?.Split('x');
            if (parts is not [var w, var h])
            {
                return resolution;
            }
            return (w.Trim(), h.Trim()) switch
            {
                ("0", "0") => "0",
                ("0", var height) => $"{height}p",
                (var width, "0") => $"{width}px wide",
                _ => resolution,
            };
        }

        private string ResolutionRange => RangeText(ResolutionText(MinResolution), ResolutionText(MaxResolution), "any resolution");

        // "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors" -> "2x IllustrationJaNai_V3denoise_FDAT_M"
        public static string ShortModelName(string? fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                return "no model";
            }
            if (fileName == "No Model")
            {
                return "No Model";
            }
            var name = Path.GetFileNameWithoutExtension(fileName);
            var scale = Regex.Match(name, @"^(\d+x)_");
            if (scale.Success)
            {
                name = $"{scale.Groups[1].Value} {name[scale.Length..]}";
            }
            var cut = Regex.Match(name, @"_(unshuffle|fp16|fp32|bf16|\d+k)(_|$)", RegexOptions.IgnoreCase);
            return cut.Success && cut.Index > 0 ? name[..cut.Index] : name;
        }

        private string ImageKinds => (IsColor, IsGrayscale) switch
        {
            (true, true) => "color + grayscale",
            (true, false) => "color",
            (false, true) => "grayscale",
            _ => "no images",
        };

        public string ActivationSummary =>
            $"{ResolutionRange} · {RangeText(MinScaleFactor?.ToString(), MaxScaleFactor?.ToString(), "any scale", "x")} · {ImageKinds}";

        public string UpscaleSummary
        {
            get
            {
                var parts = new List<string> { ShortModelName(ModelFilePath), $"tile {ModelTileSize}" };
                if (ResizeHeightBeforeUpscale > 0)
                {
                    parts.Add($"resize to {ResizeHeightBeforeUpscale}px tall first");
                }
                else if (ResizeWidthBeforeUpscale > 0)
                {
                    parts.Add($"resize to {ResizeWidthBeforeUpscale}px wide first");
                }
                else if (ResizeFactorBeforeUpscale is { } factor && factor != 100)
                {
                    parts.Add($"resize {factor}% first");
                }
                if (AutoAdjustLevels)
                {
                    parts.Add("auto levels");
                }
                return string.Join(" · ", parts);
            }
        }

        public string ChainSummary =>
            $"{ResolutionRange} · {ImageKinds} → {ShortModelName(ModelFilePath)}";

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

        public AvaloniaList<string> AllModels => _backendService.AllModels;

        public const string ConvertModelsUrl = "https://github.com/the-database/MangaJaNaiConverterGui/tree/main/tools/onnx-export";

        // 2.0 runs ONNX models; an empty model or "No Model" runs none.
        public bool HasOnnxModel =>
            string.IsNullOrWhiteSpace(ModelFilePath) || ModelFilePath == BackendService.NoModel ||
            ModelFilePath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);

        // 1.x's default models were PyTorch files (.pth / .safetensors); 2.0 ships their ONNX exports
        // under the same names, so a saved or imported chain moves to the .onnx file.
        // Bring a chain saved by 1.x up to 2.0: the ONNX default models, and a tile size from the
        // shorter list.
        public void Migrate()
        {
            ModelFilePath = MigrateModelName(ModelFilePath);
            ModelTileSize = MigrateTileSize(ModelTileSize);
        }

        public static string MigrateTileSize(string tile) => tile switch
        {
            "Auto (Estimate)" or "256" or "384" or "512" or "768" or "1024" => tile,
            "128" or "192" => "256",
            "Maximum" or "No Tiling" or "2048" or "4096" => "1024",
            _ => "Auto (Estimate)",
        };

        public static string MigrateModelName(string model)
        {
            if (string.IsNullOrWhiteSpace(model) || model.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                return model;
            }
            var ext = Path.GetExtension(model);
            return Regex.IsMatch(Path.GetFileName(model), @"^\dx_(MangaJaNai|IllustrationJaNai)_", RegexOptions.IgnoreCase) &&
                   (ext.Equals(".pth", StringComparison.OrdinalIgnoreCase) || ext.Equals(".safetensors", StringComparison.OrdinalIgnoreCase))
                ? Path.ChangeExtension(model, ".onnx")
                : model;
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

    // One entry of mangajanai_upscale --list-devices.
    public class BackendDevice
    {
        [JsonProperty("id")]
        public string Id { get; set; } = "";

        [JsonProperty("name")]
        public string Name { get; set; } = "";

        [JsonProperty("vram")]
        public long Vram { get; set; }

        // The device list shows every device the backend lists. Devices that share a name (two of the
        // same card) get the end of their id, just long enough to tell them apart: "RTX 5090 (#2600)".
        public static string[] Labels(IReadOnlyList<BackendDevice> devices) =>
            devices.Select(d =>
            {
                var twins = devices.Where(o => o.Name == d.Name).ToList();
                if (twins.Count < 2)
                {
                    return d.Name;
                }
                var longest = twins.Max(o => o.Id.Length);
                var n = Math.Min(4, longest);
                while (n < longest && twins.Select(o => Tail(o.Id, n)).Distinct().Count() < twins.Count)
                {
                    n++;
                }
                return $"{d.Name} (#{Tail(d.Id, n)})";
            }).ToArray();

        private static string Tail(string id, int n) => id.Length <= n ? id : id[^n..];

        // The saved device in a fresh list: by id; if that's gone, the device at the saved position
        // when it has the saved name (which tells identical cards apart), else the first with that
        // name; else the first device.
        public static int Find(IReadOnlyList<BackendDevice> devices, string? id, string? name, int index)
        {
            for (var i = 0; i < devices.Count; i++)
            {
                if (devices[i].Id == id)
                {
                    return i;
                }
            }
            if (index >= 0 && index < devices.Count && devices[index].Name == name)
            {
                return index;
            }
            for (var i = 0; i < devices.Count; i++)
            {
                if (devices[i].Name == name)
                {
                    return i;
                }
            }
            return 0;
        }
    }
}

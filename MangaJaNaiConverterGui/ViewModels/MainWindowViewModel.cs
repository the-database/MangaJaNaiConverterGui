using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.ViewModels
{
    [DataContract]
    public class MainWindowViewModel : ViewModelBase
    {
        public MainWindowViewModel() 
        {
            this.WhenAnyValue(x => x.InputFilePath, x => x.OutputFilePath,
                                x => x.InputFolderPath, x => x.OutputFolderPath,
                                x => x.SelectedTabIndex).Subscribe(x =>
                                {
                                    Validate();
                                });
        }

        private CancellationTokenSource? _cancellationTokenSource;
        private Process? _runningProcess = null;

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
                }
            }
        }

        private string _inputFilePath = string.Empty;
        [DataMember]
        public string InputFilePath
        {
            get => _inputFilePath;
            set => this.RaiseAndSetIfChanged(ref _inputFilePath, value);
        }

        private string _inputFolderPath = string.Empty;
        [DataMember]
        public string InputFolderPath
        {
            get => _inputFolderPath;
            set => this.RaiseAndSetIfChanged(ref _inputFolderPath, value);
        }

        private string _outputFilePath = string.Empty;
        [DataMember]
        public string OutputFilePath
        {
            get => _outputFilePath;
            set => this.RaiseAndSetIfChanged(ref _outputFilePath, value);
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

        private string _colorModelFilePath = string.Empty;
        [DataMember]
        public string ColorModelFilePath
        {
            get => _colorModelFilePath;
            set => this.RaiseAndSetIfChanged(ref _colorModelFilePath, value);
        }

        private string _resizeHeightBeforeUpscale = 0.ToString();
        [DataMember]
        public string ResizeHeightBeforeUpscale
        {
            get => _resizeHeightBeforeUpscale;
            set => this.RaiseAndSetIfChanged(ref _resizeHeightBeforeUpscale, value);
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

        private bool _valid = false;
        [IgnoreDataMember]
        public bool Valid
        {
            get => _valid;
            set
            {
                this.RaiseAndSetIfChanged(ref _valid, value);
                this.RaisePropertyChanged(nameof(UpscaleEnabled));
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
            }
        }

        private string _validationText = string.Empty;
        public string ValidationText
        {
            get => _validationText;
            set
            {
                this.RaiseAndSetIfChanged(ref _validationText, value);
            }
        }

        private string _consoleText = string.Empty;
        public string ConsoleText
        {
            get => _consoleText;
            set
            {
                this.RaiseAndSetIfChanged(ref _consoleText, value);
            }
        }

        public bool UpscaleEnabled => Valid && !Upscaling;

        public async Task RunUpscale()
        {
            // TODO use embedded python
            _cancellationTokenSource = new CancellationTokenSource();
            var ct = _cancellationTokenSource.Token;

            var task = Task.Run(async () =>
            {
                ct.ThrowIfCancellationRequested();
                ConsoleText = "";
                Upscaling = true;

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

                var imageFormat = WebpSelected ? "webp" : PngSelected ? "png" : "jpeg";

                var cmd = $@"python "".\backend\src\runmangajanaiconverterguiupscale.py"" --input-file ""{InputFilePath}"" --output-file ""{OutputFilePath}"" --input-folder ""{InputFolderPath}"" --output-folder ""{OutputFolderPath}"" --grayscale-model-path ""{GrayscaleModelFilePath}"" --color-model-path ""{ColorModelFilePath}"" --image-format {imageFormat} --lossy-compression-quality {LossyCompressionQuality} {flags}";
                ConsoleText += $"Upscaling with command: {cmd}\n";
                await RunCommand($@" /C {cmd}");

                Valid = true;
            }, ct);

            try
            {
                await task;
            }
            catch (OperationCanceledException e)
            {
                Console.WriteLine($"{nameof(OperationCanceledException)} thrown with message: {e.Message}");
                Upscaling = false;
            }
            finally
            {
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

        public void Validate()
        {
            var valid = true;
            var validationText = new List<string>();
            if (SelectedTabIndex == 0)
            {
                if (!File.Exists(InputFilePath))
                {
                    valid = false;
                    validationText.Add("Input File is required.");
                }

                if (string.IsNullOrWhiteSpace(OutputFilePath))
                {
                    valid = false;
                    validationText.Add("Output File is required.");
                }
            }
            else
            {
                if (!Directory.Exists(InputFolderPath))
                {
                    valid = false;
                    validationText.Add("Input Folder is required.");
                }

                if (string.IsNullOrWhiteSpace(OutputFolderPath))
                {
                    valid = false;
                    validationText.Add("Output Folder is required.");
                }
            }

            Valid = valid;
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

                // Create a StreamWriter to write the output to a log file
                using (var outputFile = new StreamWriter("error.log", append: true))
                {
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            outputFile.WriteLine(e.Data); // Write the output to the log file
                            ConsoleText += e.Data + "\n";
                        }
                    };

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            outputFile.WriteLine(e.Data); // Write the output to the log file
                            ConsoleText += e.Data + "\n";
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine(); // Start asynchronous reading of the output
                    await process.WaitForExitAsync();
                }
                
            }
        }
    }
}
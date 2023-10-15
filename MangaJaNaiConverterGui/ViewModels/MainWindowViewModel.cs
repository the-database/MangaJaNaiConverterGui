using ReactiveUI;
using System.Runtime.Serialization;

namespace MangaJaNaiConverterGui.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {

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
    }
}
using MangaJaNaiConverterGui.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.ReactiveUI;
using ReactiveUI;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MangaJaNaiConverterGui.Views
{
    public partial class MainWindow : ReactiveWindow<MainWindowViewModel>
    {
        private bool _autoScrollConsole = true;

        public MainWindow()
        {
            AvaloniaXamlLoader.Load(this);
            this.WhenActivated(disposable => { });
            Resized += MainWindow_Resized;
            Closing += MainWindow_Closing;

            var inputFileNameTextBox = this.FindControl<TextBox>("InputFileNameTextBox");
            var outputFileNameTextBox = this.FindControl<TextBox>("OutputFileNameTextBox");
            var inputFolderNameTextBox = this.FindControl<TextBox>("InputFolderNameTextBox");
            var outputFolderNameTextBox = this.FindControl<TextBox>("OutputFolderNameTextBox");
            var grayscaleModelFilePathTextBox = this.FindControl<TextBox>("GrayscaleModelFilePathTextBox");
            var colorModelFilePathTextBox = this.FindControl<TextBox>("ColorModelFilePathTextBox");

            inputFileNameTextBox?.AddHandler(DragDrop.DropEvent, SetInputFilePath);
            inputFolderNameTextBox?.AddHandler(DragDrop.DropEvent, SetInputFolderPath);
            outputFolderNameTextBox?.AddHandler(DragDrop.DropEvent, SetOutputFolderPath);
            grayscaleModelFilePathTextBox?.AddHandler(DragDrop.DropEvent, SetGrayscaleModelFilePath);
            colorModelFilePathTextBox?.AddHandler(DragDrop.DropEvent, SetColorModelFilePath);
        }

        private void MainWindow_Closing(object? sender, WindowClosingEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                vm.CancelUpscale();
            }
        }

        private void ConsoleScrollViewer_PropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
        {
            if (e.Property.Name == "Offset")
            {
                var consoleScrollViewer = this.FindControl<ScrollViewer>("ConsoleScrollViewer");

                if (e.NewValue is Vector newVector)
                {
                    _autoScrollConsole = newVector.Y == consoleScrollViewer?.ScrollBarMaximum.Y;
                }
            }

        }

        private void ConsoleTextBlock_PropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
        {
            if (e.Property.Name == "Text")
            {
                var consoleScrollViewer = this.FindControl<ScrollViewer>("ConsoleScrollViewer");
                if (consoleScrollViewer != null)
                {
                    if (_autoScrollConsole)
                    {
                        consoleScrollViewer.ScrollToEnd();
                    }
                }
            }
        }

        private void MainWindow_Resized(object? sender, WindowResizedEventArgs e)
        {
            // Set the ScrollViewer width based on the new parent window's width
            var consoleScrollViewer = this.FindControl<ScrollViewer>("ConsoleScrollViewer");
            if (consoleScrollViewer != null)
            {
                consoleScrollViewer.Width = Width - 40; // Adjust the width as needed
            }

        }

        private async void OpenInputFileButtonClick(object? sender, RoutedEventArgs e)
        {
            // Get top level from the current control. Alternatively, you can use Window reference instead.
            var topLevel = TopLevel.GetTopLevel(this);

            // Start async operation to open the dialog.
            var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Open Video File",
                AllowMultiple = false
            });

            if (files.Count >= 1)
            {
                if (DataContext is MainWindowViewModel vm)
                {
                    vm.InputFilePath = files[0].TryGetLocalPath() ?? "";
                }
            }
        }

        public void SetInputFilePath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = e.Data.GetFiles().ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (File.Exists(filePath))
                    {
                        vm.InputFilePath = filePath;
                    }
                }
            }
        }

        public void SetInputFolderPath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = e.Data.GetFiles().ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (Directory.Exists(filePath))
                    {
                        vm.InputFolderPath = filePath;
                    }
                }
            }
        }

        public void SetOutputFolderPath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = e.Data.GetFiles().ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (Directory.Exists(filePath))
                    {
                        vm.OutputFolderPath = filePath;
                    }
                }
            }
        }

        public void SetGrayscaleModelFilePath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = e.Data.GetFiles().ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (File.Exists(filePath))
                    {
                        vm.GrayscaleModelFilePath = filePath;
                    }
                }
            }
        }

        public void SetColorModelFilePath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = e.Data.GetFiles().ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (File.Exists(filePath))
                    {
                        vm.ColorModelFilePath = filePath;
                    }
                }
            }
        }

        private async void OpenInputFolderButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = GetTopLevel(this);

                var storageProvider = topLevel.StorageProvider;

                IStorageFolder? suggestedStartLocation = null;

                if (Directory.Exists(vm.InputFolderPath))
                {
                    suggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(Path.GetFullPath(vm.InputFolderPath)));
                }

                // Start async operation to open the dialog.
                var files = await storageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
                {
                    Title = "Open Folder",
                    AllowMultiple = false,
                    SuggestedStartLocation = suggestedStartLocation
                });

                if (files.Count >= 1)
                {
                    vm.InputFolderPath = files[0].TryGetLocalPath() ?? "";   
                }
            }
        }

        private async void OpenOutputFolderButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = GetTopLevel(this);

                var storageProvider = topLevel.StorageProvider;

                IStorageFolder? suggestedStartLocation = null;

                if (Directory.Exists(vm.OutputFolderPath))
                {
                    suggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(Path.GetFullPath(vm.OutputFolderPath)));
                }

                // Start async operation to open the dialog.
                var files = await storageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
                {
                    Title = "Open Folder",
                    AllowMultiple = false,
                    SuggestedStartLocation = suggestedStartLocation
                });

                if (files.Count >= 1)
                {
                    vm.OutputFolderPath = files[0].TryGetLocalPath() ?? "";
                }
            }
        }

        private async void OpenGrayscaleModelFileButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = GetTopLevel(this);

                var storageProvider = topLevel.StorageProvider;

                var folderPath = @".\chaiNNer\models";

                if (Directory.Exists(Path.GetDirectoryName(vm.GrayscaleModelFilePath)))
                {
                    folderPath = Path.GetDirectoryName(vm.GrayscaleModelFilePath);
                }

                // Start async operation to open the dialog.
                var files = await storageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
                {
                    Title = "Open Model File for Grayscale Images",
                    AllowMultiple = false,
                    FileTypeFilter = new FilePickerFileType[]
                    {
                    new("Model File") { Patterns = new[] { "*.pth", "*.pt", "*.ckpt", "*.onnx" }, MimeTypes = new[] { "*/*" } }, FilePickerFileTypes.All,
                    },
                    SuggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(Path.GetFullPath(folderPath))),
                });

                if (files.Count >= 1)
                {
                    vm.GrayscaleModelFilePath = files[0].TryGetLocalPath() ?? "";
                }
            }
        }

        private async void OpenColorModelFileButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = TopLevel.GetTopLevel(this);
                var storageProvider = topLevel.StorageProvider;

                var folderPath = @".\chaiNNer\models";

                if (Directory.Exists(Path.GetDirectoryName(vm.ColorModelFilePath)))
                {
                    folderPath = Path.GetDirectoryName(vm.ColorModelFilePath);
                }

                // Start async operation to open the dialog.
                var files = await storageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
                {
                    Title = "Open Model File for Color Images",
                    AllowMultiple = false,
                    FileTypeFilter = new FilePickerFileType[]
                    {
                    new("Model File") { Patterns = new[] { "*.pth", "*.pt", "*.ckpt", "*.onnx" }, MimeTypes = new[] { "*/*" } }, FilePickerFileTypes.All,
                    },
                    SuggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(Path.GetFullPath(folderPath))),
                });

                if (files.Count >= 1)
                {
                    vm.ColorModelFilePath = files[0].TryGetLocalPath() ?? "";
                }
            }
        }
    }
}
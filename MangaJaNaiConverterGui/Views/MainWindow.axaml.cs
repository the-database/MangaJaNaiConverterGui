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
using System.Threading.Tasks;
using FluentAvalonia.UI.Windowing;
using Squirrel;
using Avalonia.Layout;
using Avalonia.Media;
using Material.Icons.Avalonia;

namespace MangaJaNaiConverterGui.Views
{
    public partial class MainWindow : AppWindow
    {
        private bool _autoScrollConsole = true;
        private bool _userWantsToQuit = false;

        public MainWindow()
        {
            AvaloniaXamlLoader.Load(this);
            //this.WhenActivated(disposable => { });
            Resized += MainWindow_Resized;
            Closing += MainWindow_Closing;
            Opened += MainWindow_Opened;

            //this.WhenActivated(action => action(ViewModel!.ShowDialog.RegisterHandler(DoShowDialogAsync)));//TODO



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

        private void MainWindow_Opened(object? sender, EventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                vm.CheckAndExtractBackend();
            }
        }

        private async void MainWindow_Closing(object? sender, WindowClosingEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Show confirmation dialog
                if (!_userWantsToQuit && vm.Upscaling)
                {
                    // Cancel close to show dialog
                    e.Cancel = true;

                    _userWantsToQuit = await ShowConfirmationDialog("If you exit now, all unfinished upscales will be canceled. Are you sure you want to exit?");

                    // Close if the user confirmed
                    if (_userWantsToQuit)
                    {
                        vm.CancelUpscale();
                        Close();
                    }
                }
                else
                {
                    using var mgr = new UpdateManager("https://github.com/the-database/MangaJaNaiConverterGui/releases");
                    if (!mgr.IsInstalledApp)
                    {
                        return;
                    }
                    var newVersion = await mgr.UpdateApp();
                }
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

        private async void OpenInputFileButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = TopLevel.GetTopLevel(this);

                var storageProvider = topLevel.StorageProvider;

                IStorageFolder? suggestedStartLocation = null;

                var inputFolder = Path.GetDirectoryName(vm.InputFilePath);

                if (Directory.Exists(inputFolder))
                {
                    suggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(inputFolder));
                }

                // Start async operation to open the dialog.
                var files = await storageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
                {
                    Title = "Open Image or Archive File",
                    AllowMultiple = false,
                    FileTypeFilter = new FilePickerFileType[]
                    {
                        new("Image or Archive File") { Patterns = MainWindowViewModel.IMAGE_EXTENSIONS.Concat(MainWindowViewModel.ARCHIVE_EXTENSIONS).Select(x => $"*{x}").ToArray(),
                            MimeTypes = new[] { "*/*" } }, FilePickerFileTypes.All,
                    },
                    SuggestedStartLocation = suggestedStartLocation,
                });

                if (files.Count >= 1)
                {
                    vm.InputFilePath = files[0].TryGetLocalPath() ?? "";
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

        private async Task<bool> ShowConfirmationDialog(string message)
        {
            var dialog = new Window
            {
                Title = "Cancel unfinished upscales?",
                Width = 480,
                Height = 160,
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                //Icon = Icon, // TODO
                CanResize = false,
                ShowInTaskbar = false
            };

            var textBlock = new TextBlock
            {
                Text = message,
                Margin = new Thickness(20),
                TextWrapping = TextWrapping.Wrap,
                VerticalAlignment = VerticalAlignment.Center,
                Width = 380,
            };

            var materialIcon = new MaterialIcon
            {
                Kind = Material.Icons.MaterialIconKind.QuestionMarkCircleOutline,
                Width = 48,
                Height = 48,
            };

            var textPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Margin = new Thickness(20),
                Children = { materialIcon, textBlock },
            };

            var yesButton = new Button
            {
                Content = "Yes",
                Width = 100,
                HorizontalAlignment = HorizontalAlignment.Center,
                HorizontalContentAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                VerticalContentAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 0)
            };
            yesButton.Click += (sender, e) => dialog.Close(true);

            var noButton = new Button
            {
                Content = "No",
                Width = 100,
                HorizontalAlignment = HorizontalAlignment.Center,
                HorizontalContentAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                VerticalContentAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 0)
            };
            noButton.Click += (sender, e) => dialog.Close(false);

            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Children = { yesButton, noButton },
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(20, 0, 20, 20)
            };

            var mainPanel = new StackPanel
            {
                Children = { textPanel, buttonPanel }
            };

            dialog.Content = mainPanel;
            var result = await dialog.ShowDialog<bool?>(this);

            return result ?? false;
        }
    }
}
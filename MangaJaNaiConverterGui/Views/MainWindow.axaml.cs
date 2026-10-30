using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using FluentAvalonia.UI.Controls;
using FluentAvalonia.UI.Windowing;
using MangaJaNaiConverterGui.ViewModels;
using Material.Icons.Avalonia;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.Views
{
    public partial class MainWindow : FAAppWindow
    {
        // Consoles the user scrolled up in; the others follow new output.
        private readonly HashSet<ScrollViewer> _scrolledUpConsoles = [];
        private bool _userWantsToQuit = false;

        public MainWindow()
        {
            AvaloniaXamlLoader.Load(this);

            Closing += MainWindow_Closing;
            // Installs a staged update once the app has exited.
            Closed += (_, _) => (DataContext as MainWindowViewModel)?.OnAppExit();
            Opened += MainWindow_Opened;

            var inputFileNameTextBox = this.FindControl<TextBox>("InputFileNameTextBox");
            var inputFolderNameTextBox = this.FindControl<TextBox>("InputFolderNameTextBox");
            var outputFolderNameTextBox = this.FindControl<TextBox>("OutputFolderNameTextBox");

            inputFileNameTextBox?.AddHandler(DragDrop.DropEvent, SetInputFilePath);
            inputFolderNameTextBox?.AddHandler(DragDrop.DropEvent, SetInputFolderPath);
            outputFolderNameTextBox?.AddHandler(DragDrop.DropEvent, SetOutputFolderPath);
        }

        private async void MainWindow_Opened(object? sender, EventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var sidebar = this.FindControl<Grid>("MainGrid")?.ColumnDefinitions[0];
                if (sidebar != null && vm.SidebarWidth > 0)
                {
                    sidebar.Width = new GridLength(Math.Clamp(vm.SidebarWidth, sidebar.MinWidth, sidebar.MaxWidth));
                }

                vm.CheckAndExtractBackend();
            }
        }

        private void SidebarSplitterDragCompleted(object? sender, VectorEventArgs e)
        {
            var sidebar = this.FindControl<Grid>("MainGrid")?.ColumnDefinitions[0];
            if (DataContext is MainWindowViewModel vm && sidebar != null)
            {
                vm.SidebarWidth = sidebar.ActualWidth;
            }
        }

        // Narrower than this, the sidebar shows only icons.
        private const double CompactSidebarWidth = 120;

        private void SidebarSizeChanged(object? sender, SizeChangedEventArgs e)
        {
            if (sender is Control sidebar)
            {
                sidebar.Classes.Set("compact", e.NewSize.Width < CompactSidebarWidth);
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

                    _userWantsToQuit = await ShowConfirmationDialog("Cancel unfinished upscales?", "If you exit now, all unfinished upscales will be canceled. Are you sure you want to exit?");

                    // Close if the user confirmed
                    if (_userWantsToQuit)
                    {
                        vm.CancelUpscale();
                        Close();
                    }
                }
                else
                {
                }
            }
        }

        private void ConsoleScrollViewer_PropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
        {
            if (e.Property.Name == "Offset" && sender is ScrollViewer consoleScrollViewer && e.NewValue is Vector newVector)
            {
                if (newVector.Y >= consoleScrollViewer.ScrollBarMaximum.Y)
                {
                    _scrolledUpConsoles.Remove(consoleScrollViewer);
                }
                else
                {
                    _scrolledUpConsoles.Add(consoleScrollViewer);
                }
            }
        }

        private void ConsoleTextBlock_PropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
        {
            if (e.Property.Name == "Text" && sender is TextBlock textBlock && textBlock.Parent is ScrollViewer consoleScrollViewer
                && !_scrolledUpConsoles.Contains(consoleScrollViewer))
            {
                consoleScrollViewer.ScrollToEnd();
            }
        }

        // Dragging the handle above the console resizes it; the height is remembered.
        private void ConsoleResizeDragDelta(object? sender, VectorEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var max = Math.Max(100, Bounds.Height - 250);
                vm.ConsoleHeight = Math.Clamp(vm.ConsoleHeight - e.Vector.Y, 100, max);
            }
        }

        public void SetInputFilePath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = (e.DataTransfer.TryGetFiles() ?? []).ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (File.Exists(filePath))
                    {
                        vm.CurrentWorkflow.InputFilePath = filePath;
                    }
                }
            }
        }

        public void SetInputFolderPath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = (e.DataTransfer.TryGetFiles() ?? []).ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (Directory.Exists(filePath))
                    {
                        vm.CurrentWorkflow.InputFolderPath = filePath;
                    }
                }
            }
        }

        public void SetOutputFolderPath(object? sender, DragEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                var files = (e.DataTransfer.TryGetFiles() ?? []).ToList();


                if (files.Count > 0)
                {
                    var filePath = files[0].TryGetLocalPath();
                    if (Directory.Exists(filePath))
                    {
                        vm.CurrentWorkflow.OutputFolderPath = filePath;
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

                var inputFolder = Path.GetDirectoryName(vm.CurrentWorkflow.InputFilePath);

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
                    vm.CurrentWorkflow.InputFilePath = files[0].TryGetLocalPath() ?? "";
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

                if (Directory.Exists(vm.CurrentWorkflow.InputFolderPath))
                {
                    suggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(Path.GetFullPath(vm.CurrentWorkflow.InputFolderPath)));
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
                    vm.CurrentWorkflow.InputFolderPath = files[0].TryGetLocalPath() ?? "";
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

                if (Directory.Exists(vm.CurrentWorkflow.OutputFolderPath))
                {
                    suggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(new Uri(Path.GetFullPath(vm.CurrentWorkflow.OutputFolderPath)));
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
                    vm.CurrentWorkflow.OutputFolderPath = files[0].TryGetLocalPath() ?? "";
                }
            }
        }

        private async void ImportCurrentWorkflowButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = GetTopLevel(this);

                // Start async operation to open the dialog.
                var storageProvider = topLevel.StorageProvider;

                var files = await storageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
                {
                    Title = "Import Workflow File",
                    AllowMultiple = false,
                    FileTypeFilter = [new("MangaJaNai Workflow File") { Patterns = ["*.mwf"], MimeTypes = ["*/*"] }, FilePickerFileTypes.All],
                    SuggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)),
                });

                if (files.Count >= 1)
                {

                    var inPath = files[0].TryGetLocalPath();

                    if (inPath != null)
                    {
                        var td = new FATaskDialog
                        {
                            Title = "Confirm Workflow Import",
                            ShowProgressBar = false,
                            Content = $"The following workflow file will be imported to the current workflow {vm.CurrentWorkflow?.WorkflowName}. All configuration settings for the current profile {vm.CurrentWorkflow?.WorkflowName} will be overwritten.\n\n" +
                            inPath,
                            Buttons =
                        {
                            FATaskDialogButton.OKButton,
                            FATaskDialogButton.CancelButton
                        }
                        };

                        // The XamlRoot must be the window: in Avalonia 12 VisualRoot is an internal host
                        // that FATaskDialog can't show on.
                        td.XamlRoot = this;
                        // Swap the workflow on the UI thread: it's bound to the sidebar and the form.
                        if (await td.ShowAsync() is FATaskDialogStandardResult.OK)
                        {
                            vm.ReadWorkflowFileToCurrentWorkflow(inPath);
                        }
                    }
                }
            }
        }

        private async void ResetWorkflow(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {

                var td = new FATaskDialog
                {
                    Title = "Confirm Workflow Reset",
                    ShowProgressBar = false,
                    Content = $"The current workflow's settings will be reset to the default settings. Any unsaved settings for the current workflow will be lost.",
                    Buttons =
                {
                    FATaskDialogButton.OKButton,
                    FATaskDialogButton.CancelButton
                }
                };

                td.XamlRoot = this;
                if (await td.ShowAsync() is FATaskDialogStandardResult.OK)
                {
                    vm.ResetCurrentWorkflow();
                }

            }
        }

        private async void ExportCurrentWorkflowButtonClick(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
            {
                // Get top level from the current control. Alternatively, you can use Window reference instead.
                var topLevel = GetTopLevel(this);

                var storageProvider = topLevel.StorageProvider;

                // Start async operation to open the dialog.
                var file = await storageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                {
                    Title = "Export Workflow File",
                    DefaultExtension = "mwf",
                    FileTypeChoices =
                    [
                    new("MangaJaNai Workflow File (*.mwf)") { Patterns = ["*.mwf"] },
                    ],
                    SuggestedStartLocation = await storageProvider.TryGetFolderFromPathAsync(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)),
                    SuggestedFileName = vm.CurrentWorkflow?.WorkflowName,
                });

                if (file is not null)
                {
                    var outPath = file.TryGetLocalPath();

                    if (outPath != null)
                    {
                        vm.WriteCurrentWorkflowToFile(outPath);
                    }
                }
            }
        }

        private async Task<bool> ShowConfirmationDialog(string title, string message)
        {
            var dialog = new Window
            {
                Title = title,
                Width = 480,
                Height = 200,
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

        public void HandleDevicesGotFocus(object? sender, FocusChangedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm && sender is AutoCompleteBox cb)
            {
                if (vm.CurrentWorkflow != null && e.NavigationMethod != NavigationMethod.Unspecified && e.Source is TextBox)
                {
                    cb.IsDropDownOpen = true;
                }
            }
        }

        private void HandleDevicesTapped(object? sender, Avalonia.Input.TappedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm && sender is AutoCompleteBox cb)
            {
                if (vm.CurrentWorkflow != null)
                {
                    cb.IsDropDownOpen = true;
                }
            }
        }
    }
}
using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using MangaJaNaiConverterGui.ViewModels;

namespace MangaJaNaiConverterGui;

public partial class SettingsWindow : ReactiveWindow<MainWindowViewModel>
{
    public SettingsWindow()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
using Avalonia;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using ReactiveUI;
using System;
using System.IO;
using System.Reflection;

namespace MangaJaNaiConverterGui
{
    public partial class App : Application
    {
        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);
        }

        public override void OnFrameworkInitializationCompleted()
        {
            var appStateFolder = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), 
                "MangaJaNaiConverterGui"
            );
            var appStatePath = Path.Combine(appStateFolder, "appstate.json");

            if (!Directory.Exists(appStateFolder))
            {
                Directory.CreateDirectory(appStateFolder);
            }

            if (!File.Exists(appStatePath))
            {
                File.Copy("appstate.json", appStatePath);
            }

            var suspension = new AutoSuspendHelper(ApplicationLifetime);
            RxApp.SuspensionHost.CreateNewAppState = () => new MainWindowViewModel();
            RxApp.SuspensionHost.SetupDefaultSuspendResume(new NewtonsoftJsonSuspensionDriver(appStatePath));
            suspension.OnFrameworkInitializationCompleted();
            
            // Load the saved view model state.
            var state = RxApp.SuspensionHost.GetAppState<MainWindowViewModel>();
            new MainWindow { DataContext = state }.Show();
            base.OnFrameworkInitializationCompleted();
        }
    }
}
using Avalonia;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using ReactiveUI;
using System;
using System.IO;
using System.Reactive.Linq;
using System.Reactive;
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
            if (!Directory.Exists(Program.AppStateFolder))
            {
                Directory.CreateDirectory(Program.AppStateFolder);
            }

            if (!File.Exists(Program.AppStatePath))
            {
                File.Copy(Program.AppStateFilename, Program.AppStatePath);
            }

            var suspension = new AutoSuspendHelper(ApplicationLifetime);
            RxApp.SuspensionHost.CreateNewAppState = () => new MainWindowViewModel();
            //var suspensionDriver = new NewtonsoftJsonSuspensionDriver(Program.AppStatePath);
            RxApp.SuspensionHost.SetupDefaultSuspendResume(Program.SuspensionDriver);
            suspension.OnFrameworkInitializationCompleted();            

            // Load the saved view model state.
            var state = RxApp.SuspensionHost.GetAppState<MainWindowViewModel>();
            foreach (var wf in state.Workflows)
            {
                wf.Vm = state;
            }

            state.CurrentWorkflow?.Validate();

            new MainWindow { DataContext = state }.Show();
            base.OnFrameworkInitializationCompleted();
        }
    }
}
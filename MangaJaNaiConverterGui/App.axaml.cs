using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using NuGet.Versioning;
using ReactiveUI;
using Squirrel;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui
{
    public partial class App : Application
    {
        public override void Initialize()
        {
            SquirrelAwareApp.HandleEvents(
                onInitialInstall: OnAppInstall,
                onAppUninstall: OnAppUninstall
            );
            AvaloniaXamlLoader.Load(this);
        }

        public override async void OnFrameworkInitializationCompleted()
        {
            var suspension = new AutoSuspendHelper(ApplicationLifetime);
            RxApp.SuspensionHost.CreateNewAppState = () => new MainWindowViewModel();
            RxApp.SuspensionHost.SetupDefaultSuspendResume(new NewtonsoftJsonSuspensionDriver("appstate.json"));
            suspension.OnFrameworkInitializationCompleted();
            
            // Load the saved view model state.
            var state = RxApp.SuspensionHost.GetAppState<MainWindowViewModel>();
            new MainWindow { DataContext = state }.Show();
            base.OnFrameworkInitializationCompleted();
        }

        private static void OnAppInstall(SemanticVersion version, IAppTools tools)
        {
            tools.CreateShortcutForThisExe(ShortcutLocation.StartMenu | ShortcutLocation.Desktop);
        }

        private static void OnAppUninstall(SemanticVersion version, IAppTools tools)
        {
            tools.RemoveShortcutForThisExe(ShortcutLocation.StartMenu | ShortcutLocation.Desktop);
        }
    }
}
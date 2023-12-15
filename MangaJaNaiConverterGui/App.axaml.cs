using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using ReactiveUI;
using Squirrel;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui
{
    public partial class App : Application
    {
        public override void Initialize()
        {
            SquirrelAwareApp.HandleEvents();
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
            if (state.AutoUpdateEnabled)
            {
                await UpdateMyApp();
            }
            new MainWindow { DataContext = state }.Show();
            base.OnFrameworkInitializationCompleted();
        }

        private static async Task UpdateMyApp()
        {
            using var mgr = new UpdateManager("https://github.com/the-database/MangaJaNaiConverterGui/releases");
            var newVersion = await mgr.UpdateApp();

            // You must restart to complete the update. 
            // This can be done later / at any time.
            if (newVersion != null) UpdateManager.RestartApp();
        }
    }
}
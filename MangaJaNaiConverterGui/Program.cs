using Avalonia;
using ReactiveUI.Avalonia;
using System;

namespace MangaJaNaiConverterGui
{
    internal class Program
    {
        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args)
        {
            // Velopack removed: install/uninstall/update are handled by the Inno installer +
            // MangaJaNaiUpdater. Uninstall cleanup is the installer's [UninstallDelete].
            BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        }

        // Avalonia configuration, don't remove; also used by visual designer.
        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .WithInterFont()
                .LogToTrace()
                // UseReactiveUI runs WithAvalonia() (which configures a SuspensionHost<Unit>) and then
                // invokes this callback before committing the suspension host singleton. We override it
                // with a plain non-generic SuspensionHost (AppState starts null, like the old
                // RxApp.SuspensionHost) so the app state can be loaded/typed as MainWindowViewModel.
                // Otherwise GetAppState<MainWindowViewModel>() in App throws InvalidCastException.
                .UseReactiveUI(b => b.WithSuspensionHost());
    }
}

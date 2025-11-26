using Avalonia;
using ReactiveUI.Avalonia;
using System;
using System.IO;
using Velopack;

namespace MangaJaNaiConverterGui
{
    internal class Program
    {
        public static bool WasFirstRun { get; private set; }

        public static readonly string InstalledAppStateFolder = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "MangaJaNaiConverterGui"
        );

        public static readonly string InstalledAppStateFilename = "appstate2.json";
        public static readonly string InstalledAppStatePath = Path.Combine(InstalledAppStateFolder, InstalledAppStateFilename);

        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args)
        {
            VelopackApp.Build()
                .OnBeforeUninstallFastCallback((v) =>
                {
                    // On uninstall, remove Python and models from app data
                    var pythonDir = Path.Combine(InstalledAppStateFolder, "python");
                    var modelsDir = Path.Combine(InstalledAppStateFolder, "models");
                    if (Directory.Exists(pythonDir))
                    {
                        Directory.Delete(pythonDir, true);
                    }
                    if (Directory.Exists(modelsDir))
                    {
                        Directory.Delete(modelsDir, true);
                    }
                })
                .OnFirstRun(_ =>
                {
                    WasFirstRun = true;
                })
                .Run();
            BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        }

        // Avalonia configuration, don't remove; also used by visual designer.
        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .WithInterFont()
                .LogToTrace()
                .UseReactiveUI();
    }
}
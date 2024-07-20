using Avalonia;
using Avalonia.ReactiveUI;
using NuGet.Versioning;
using Velopack;
using System;
using Microsoft.Extensions.Logging;
using System.IO;
using ReactiveUI;
using MangaJaNaiConverterGui.Drivers;

namespace MangaJaNaiConverterGui
{
    internal class Program
    {
        public static readonly string AppStateFolder = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "MangaJaNaiConverterGui"
        );
        public static readonly string AppStateFilename = "appstate2.json";
        public static readonly string AppStatePath = Path.Combine(AppStateFolder, AppStateFilename);

        public static readonly ISuspensionDriver SuspensionDriver = new NewtonsoftJsonSuspensionDriver(AppStatePath);

        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args)
        {
            VelopackApp.Build()
                .WithBeforeUninstallFastCallback((v) => 
                {
                    // On uninstall, remove Python and models from app data
                    var pythonDir = Path.Combine(AppStateFolder, "python");
                    var modelsDir = Path.Combine(AppStateFolder, "models");
                    if (Directory.Exists(pythonDir))
                    { 
                        Directory.Delete(pythonDir, true);
                    }
                    if (Directory.Exists(modelsDir))
                    {
                        Directory.Delete(modelsDir, true);
                    }
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
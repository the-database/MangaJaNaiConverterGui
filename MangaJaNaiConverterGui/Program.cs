using Avalonia;
using Avalonia.ReactiveUI;
using NuGet.Versioning;
using Velopack;
using System;
using Microsoft.Extensions.Logging;
using System.IO;

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

        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args)
        {
            VelopackApp.Build().Run();
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
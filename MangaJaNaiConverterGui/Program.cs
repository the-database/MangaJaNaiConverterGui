using Avalonia;
using Avalonia.ReactiveUI;
using System;
using Velopack;
using Serilog;
using Serilog.Extensions.Logging;

namespace MangaJaNaiConverterGui
{
    internal class Program
    {
        public static Microsoft.Extensions.Logging.ILogger Log { get; private set; }

        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args)
        {

            //Log = new LoggerConfiguration().WriteTo.Console().CreateLogger();

            var serilogLogger = new LoggerConfiguration()
            .Enrich.FromLogContext()
            .MinimumLevel.Verbose()
            .WriteTo.File("mangajanaiconvertergui.log") // Serilog.Sinks.Debug
            .CreateLogger();

            var microsoftLogger = new SerilogLoggerFactory(serilogLogger)
                .CreateLogger(nameof(Program));

            VelopackApp.Build().Run(microsoftLogger);
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
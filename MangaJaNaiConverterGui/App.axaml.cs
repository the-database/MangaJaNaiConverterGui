using Autofac;
using Avalonia;
using Avalonia.Markup.Xaml;
using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using ReactiveUI;
using ReactiveUI.Avalonia;
using ReactiveUI.Builder;
using Splat;
using Splat.Autofac;
using System.IO;

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
            // Create a new Autofac container builder.
            var builder = new ContainerBuilder();
            builder.RegisterType<MainWindowViewModel>().AsSelf();
            builder.RegisterType<BackendService>().As<IBackendService>().SingleInstance();
            builder.RegisterType<UpdateManagerService>().As<IUpdateManagerService>().SingleInstance();
            builder.RegisterType<SuspensionDriverService>().As<ISuspensionDriverService>().SingleInstance();
            // etc.
            // Register the Adapter to Splat.
            // Creates and sets the Autofac resolver as the Locator.
            var autofacResolver = builder.UseAutofacDependencyResolver();

            // Register the resolver in Autofac so it can be later resolved.
            builder.RegisterInstance(autofacResolver);

            // Initialize ReactiveUI into the Autofac-backed resolver (replaces the old
            // InitializeReactiveUI() extension). Statement form, not a fluent chain: WithCoreServices()
            // returns the base IAppBuilder, which doesn't expose the full builder method set.
            var rxuiBuilder = autofacResolver.CreateReactiveUIBuilder();
            rxuiBuilder.WithPlatformServices();
            rxuiBuilder.WithAvalonia();
            rxuiBuilder.WithCoreServices();
            rxuiBuilder.Build();

            var container = builder.Build();

            autofacResolver.SetLifetimeScope(container);

            var backendService = container.Resolve<IBackendService>();

            Directory.CreateDirectory(backendService.AppStateFolder);

            // The default state ships next to the app; the working directory is wherever the app was
            // launched from.
            var defaultStatePath = NewtonsoftJsonSuspensionDriver.DefaultStatePath;
            if (!File.Exists(backendService.AppStatePath) && File.Exists(defaultStatePath)
                && !string.Equals(Path.GetFullPath(backendService.AppStatePath), Path.GetFullPath(defaultStatePath), System.StringComparison.OrdinalIgnoreCase))
            {
                File.Copy(defaultStatePath, backendService.AppStatePath);
            }

            var suspension = new AutoSuspendHelper(ApplicationLifetime!);
            // Used when the saved state can't be loaded: start from the bundled defaults (an empty view
            // model has no workflows, which the code below can't run with).
            RxSuspension.SuspensionHost.CreateNewAppState = () =>
                NewtonsoftJsonSuspensionDriver.LoadDefaultState() ?? new MainWindowViewModel();
            RxSuspension.SuspensionHost.SetupDefaultSuspendResume(container.Resolve<ISuspensionDriverService>().SuspensionDriver);
            suspension.OnFrameworkInitializationCompleted();

            // Load the saved view model state.
            var state = RxSuspension.SuspensionHost.GetAppState<MainWindowViewModel>();

            foreach (var wf in state.Workflows)
            {
                wf.Vm = state;

                foreach (var chain in wf.Chains)
                {
                    chain.Vm = state;
                    chain.Migrate();
                }
            }

            state.CurrentWorkflow?.Validate();
            state.CheckAndDoBackup();
            state.StartUpdateChecks();

            new MainWindow { DataContext = state }.Show();
            NewtonsoftJsonSuspensionDriver.StartupComplete = true;
            base.OnFrameworkInitializationCompleted();
        }
    }
}
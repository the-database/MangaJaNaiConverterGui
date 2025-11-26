using Autofac;
using Avalonia;
using Avalonia.Markup.Xaml;
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using ReactiveUI;
using Splat;
using Splat.Autofac;
using System.IO;
using ReactiveUI;
using ReactiveUI.Avalonia;
using Splat;
using Splat.Autofac;

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
            builder.RegisterType<PythonService>().As<IPythonService>().SingleInstance();
            builder.RegisterType<UpdateManagerService>().As<IUpdateManagerService>().SingleInstance();
            builder.RegisterType<SuspensionDriverService>().As<ISuspensionDriverService>().SingleInstance();
            // etc.
            // Register the Adapter to Splat.
            // Creates and sets the Autofac resolver as the Locator.
            var autofacResolver = builder.UseAutofacDependencyResolver();

            // Register the resolver in Autofac so it can be later resolved.
            builder.RegisterInstance(autofacResolver);

            // Initialize ReactiveUI components.
            autofacResolver.InitializeReactiveUI();

            var container = builder.Build();

            autofacResolver.SetLifetimeScope(container);

            //var vm = container.Resolve<MainWindowViewModel>();
            var umService = container.Resolve<IUpdateManagerService>();

            if (umService.IsInstalled)
            {
                if (!Directory.Exists(Program.InstalledAppStateFolder))
                {
                    Directory.CreateDirectory(Program.InstalledAppStateFolder);
                }

                if (!File.Exists(Program.InstalledAppStatePath))
                {
                    File.Copy(Program.InstalledAppStateFilename, Program.InstalledAppStatePath);
                }
            }

            var suspension = new AutoSuspendHelper(ApplicationLifetime);
            RxApp.SuspensionHost.CreateNewAppState = () => new MainWindowViewModel();
            RxApp.SuspensionHost.SetupDefaultSuspendResume(container.Resolve<ISuspensionDriverService>().SuspensionDriver);
            suspension.OnFrameworkInitializationCompleted();

            // Load the saved view model state.
            var state = RxApp.SuspensionHost.GetAppState<MainWindowViewModel>();

            foreach (var wf in state.Workflows)
            {
                wf.Vm = state;

                foreach (var chain in wf.Chains)
                {
                    chain.Vm = state;
                }
            }

            state.CurrentWorkflow?.Validate();

            new MainWindow { DataContext = state }.Show();
            base.OnFrameworkInitializationCompleted();
        }
    }
}
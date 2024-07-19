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
using Microsoft.Extensions.DependencyInjection;
using MangaJaNaiConverterGui.Extensions;
using Splat;
using MangaJaNaiConverterGui.Services;
using Autofac;
using Splat.Autofac;
using OctaneEngineCore;

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

            // Create a new Autofac container builder.
            var builder = new ContainerBuilder();
            builder.AddOctane();
            builder.RegisterType<MainWindowViewModel>();//.AsSelf();
            builder.RegisterType<PythonService>().As<IPythonService>();
            builder.RegisterType<UpdateManagerService>().As<IUpdateManagerService>();
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

            var suspension = new AutoSuspendHelper(ApplicationLifetime);
            RxApp.SuspensionHost.CreateNewAppState = () => new MainWindowViewModel();
            RxApp.SuspensionHost.SetupDefaultSuspendResume(Program.SuspensionDriver);
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
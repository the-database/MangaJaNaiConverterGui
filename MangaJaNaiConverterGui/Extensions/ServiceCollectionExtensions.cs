using Avalonia;
using Avalonia.ReactiveUI;
using NuGet.Versioning;
using Velopack;
using System;
using Microsoft.Extensions.Logging;
using System.IO;
using ReactiveUI;
using Microsoft.Extensions.DependencyInjection;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Services;

namespace MangaJaNaiConverterGui.Extensions
{
    public static class ServiceCollectionExtensions
    {
        public static void AddCommonServices(this IServiceCollection collection)
        {
            collection.AddTransient<IPythonService, PythonService>();
            collection.AddTransient<MainWindowViewModel>();
        }
    }
}
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

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
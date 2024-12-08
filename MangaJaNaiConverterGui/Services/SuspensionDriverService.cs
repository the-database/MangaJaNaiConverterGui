using MangaJaNaiConverterGui.Drivers;
using ReactiveUI;

namespace MangaJaNaiConverterGui.Services
{
    public class SuspensionDriverService(IPythonService pythonService) : ISuspensionDriverService
    {
        private readonly ISuspensionDriver _driver = new NewtonsoftJsonSuspensionDriver(pythonService.AppStatePath);
        public ISuspensionDriver SuspensionDriver => _driver;
    }
}

using MangaJaNaiConverterGui.Drivers;
using ReactiveUI;

namespace MangaJaNaiConverterGui.Services
{
    public class SuspensionDriverService(IBackendService backendService) : ISuspensionDriverService
    {
        private readonly ISuspensionDriver _driver = new NewtonsoftJsonSuspensionDriver(backendService.AppStatePath);
        public ISuspensionDriver SuspensionDriver => _driver;
    }
}

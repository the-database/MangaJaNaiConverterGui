using System;
using System.Threading.Tasks;
using Velopack;

namespace MangaJaNaiConverterGui.Services
{
    public interface IUpdateManagerService
    {
        bool IsInstalled { get; }
        bool IsPortable { get; }
        string AppVersion { get; }
        bool IsUpdatePendingRestart { get; }
        void ApplyUpdatesAndRestart(UpdateInfo update);
        Task<UpdateInfo?> CheckForUpdatesAsync();
        Task DownloadUpdatesAsync(UpdateInfo update, Action<int>? progress = null);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Velopack;

namespace MangaJaNaiConverterGui.Services
{
    public interface IUpdateManagerService
    {
        bool IsInstalled { get; }
        string AppVersion { get; }
        bool IsUpdatePendingRestart { get; }
        void ApplyUpdatesAndRestart(UpdateInfo update);
        Task<UpdateInfo?> CheckForUpdatesAsync();
        Task DownloadUpdatesAsync(UpdateInfo update, Action<int>? progress = null);
    }
}

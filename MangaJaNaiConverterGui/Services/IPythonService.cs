using Avalonia.Collections;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.Services
{
    public interface IPythonService
    {
        bool IsPythonInstalled();

        Task<bool> IsPythonUpdated();
        bool AreModelsInstalled();
        string BackendDirectory { get; }
        string LogsDirectory { get; }
        string PythonDirectory { get; }
        string ModelsDirectory { get; }
        string PythonPath { get; }
        string AppStateFolder { get; }
        string AppStatePath { get; }
        string AppStateFilename { get; }
        string InstallUpdatePythonDependenciesCommand { get; }
        void ExtractTgz(string gzArchiveName, string destFolder);
        void ExtractZip(string archivePath, string outFolder, ProgressChanged progressChanged);
        void AddPythonPth(string destFolder);
        AvaloniaList<string> AllModels { get; }
    }
}

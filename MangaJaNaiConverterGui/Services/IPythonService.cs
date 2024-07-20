using Avalonia.Collections;
using ICSharpCode.SharpZipLib.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.Services
{
    public interface IPythonService
    {
        bool IsPythonInstalled();
        bool AreModelsInstalled();
        string PythonDirectory { get; }
        string ModelsDirectory { get; }
        string PythonPath { get; }
        string InstallUpdatePythonDependenciesCommand { get; }
        void ExtractTgz(string gzArchiveName, string destFolder);
        void ExtractZip(string archivePath, string outFolder, ProgressChanged progressChanged);
        void AddPythonPth(string destFolder);
        AvaloniaList<string> AllModels { get; }
    }
}

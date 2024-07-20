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
        string PythonDirectory { get; }
        string PythonPath { get; }
        string InstallUpdatePythonDependenciesCommand { get; }
        void ExtractTgz(string gzArchiveName, string destFolder);
        void AddPythonPth(string destFolder);
    }
}

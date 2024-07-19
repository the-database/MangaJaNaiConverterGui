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
        Task InstallPython();
        string PythonPath { get; }
    }
}

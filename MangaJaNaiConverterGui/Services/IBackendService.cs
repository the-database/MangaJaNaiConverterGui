using Avalonia.Collections;

namespace MangaJaNaiConverterGui.Services
{
    // Where the upscaling backend (mangajanai_upscale.exe + libaji) and the app's data live.
    public interface IBackendService
    {
        string AppStateFolder { get; }
        string AppStateFilename { get; }
        string AppStatePath { get; }
        string LogsDirectory { get; }

        // The inference folder: libaji, its runtimes and mangajanai_upscale.exe.
        string InferenceDirectory { get; }
        string BackendExePath { get; }
        bool IsBackendInstalled { get; }

        // The TensorRT-RTX backend (the trt-rtx component) and the TensorRT runtime (trt-runtime),
        // which TensorRT-RTX needs as well (its CUDA runtime, and TensorRT for the fallback).
        bool IsTensorRtRtxInstalled { get; }
        bool IsTensorRtRuntimeInstalled { get; }

        // Models the user added (shown first) and the default models installed with the app.
        string UserModelsDirectory { get; }
        string BundledModelsDirectory { get; }
        bool AreModelsInstalled { get; }

        // TensorRT (*.engine) and TensorRT-RTX (*.rtxengine + *.jitcache) engines built for this GPU,
        // one per model and tile size. Each backend's stale cleanup only touches its own files.
        string EngineCacheDirectory { get; }

        // The ONNX models in both folders, plus "No Model". Cached; RefreshModels rescans.
        AvaloniaList<string> AllModels { get; }
        void RefreshModels();
    }
}

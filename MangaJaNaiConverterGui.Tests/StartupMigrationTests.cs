using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using MangaJaNaiConverterGui.Drivers;
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using MangaJaNaiConverterGui.Views;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    // A 1.x state opened by 2.0 and saved again, in the real main window: the workflow shown at
    // startup has its model pickers bound before the device list (and so the variant rule) arrives,
    // while its chains name models the pickers don't list yet (a 1.x .pth whose ONNX export is
    // installed under its _fp16 name) or at all (a 1.x model 2.0 doesn't ship). Neither name may
    // be lost: the first resolves to the _fp16 file, the second stays, shown as not installed.
    public class StartupMigrationTests
    {
        const string MangaPth = "2x_MangaJaNai_1600p_V1_ESRGAN_90k.pth";
        const string MangaFp16 = "2x_MangaJaNai_1600p_V1_ESRGAN_90k_fp16.onnx";
        const string V2 = "4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors";
        const string V2Onnx = "4x_IllustrationJaNai_V2standard_FDAT_M_52k.onnx";
        const string HatSafetensors = "4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.safetensors";
        const string HatBf16 = "4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx";
        const int Shown = 3;

        static readonly BackendDevice Rtx5090 = new() { Id = "112600", Name = "NVIDIA GeForce RTX 5090", ComputeCapability = "12.0" };

        static string[] Installed => [.. ModelVariants.ShippedModels, "No Model"];

        // The bundled default state, turned into what 1.x saved: PyTorch model names everywhere, the
        // shown workflow's chains open, naming the .pth and the V2 model.
        static string LegacyState()
        {
            var state = JObject.Parse(File.ReadAllText(NewtonsoftJsonSuspensionDriver.DefaultStatePath));
            state["SelectedWorkflowIndex"] = Shown;
            var workflows = (JArray)state["Workflows"]!["$values"]!;
            foreach (var chain in workflows.SelectMany(w => (JArray)w["Chains"]!["$values"]!))
            {
                var model = (string)chain["ModelFilePath"]!;
                chain["ModelFilePath"] = model.StartsWith("2x_MangaJaNai") || model.StartsWith("4x_MangaJaNai")
                    ? model.Replace("_fp16.onnx", ".pth")
                    : HatSafetensors;
            }
            var shown = workflows[Shown];
            shown["ShowAdvancedSettings"] = true;
            var chains = (JArray)shown["Chains"]!["$values"]!;
            while (chains.Count > 2)
            {
                chains.RemoveAt(2);
            }
            chains[0]["ModelFilePath"] = MangaPth;
            chains[1]["ModelFilePath"] = V2;
            foreach (var c in chains)
            {
                c["ChainSectionExpanded"] = true;
            }
            return state.ToString();
        }

        // What App.OnFrameworkInitializationCompleted does with the loaded state.
        static MainWindowViewModel Load(string json)
        {
            var vm = new MainWindowViewModel(new FakeBackendService(Installed), new FakeUpdateManagerService());
            JsonConvert.PopulateObject(json, vm, NewtonsoftJsonSuspensionDriver.Settings);
            foreach (var wf in vm.Workflows!)
            {
                wf.Vm = vm;
                foreach (var chain in wf.Chains)
                {
                    chain.Vm = vm;
                    chain.Migrate();
                }
            }
            vm.CurrentWorkflow?.Validate();
            return vm;
        }

        static List<ComboBox> ModelPickers(Window window) =>
            window.GetLogicalDescendants().OfType<ComboBox>().Where(c => c.DataContext is UpscaleChain).ToList();

        static string? ShownModel(ComboBox picker) => (picker.SelectedItem as ModelChoice)?.Name;

        // One test, one main window: FluentAvalonia's window can't be created twice in a process here.
        [AvaloniaFact]
        public void ALegacyStateKeepsItsModelsThroughStartupAndSave()
        {
            var vm = Load(LegacyState());
            // The window as CheckAndExtractBackend leaves it while the device list is on its way:
            // the main form shown (setup done), no devices yet. (Its Opened handler runs
            // CheckAndExtractBackend only with the view model already there.)
            var window = new MainWindow();
            window.Show();
            vm.IsExtractingBackend = false;
            window.DataContext = vm;
            Dispatcher.UIThread.RunJobs();
            window.UpdateLayout();

            var chains = vm.CurrentWorkflow!.Chains;
            var pickers = ModelPickers(window);
            Assert.Equal(2, pickers.Count);

            // Bound before the variant rule ran: the names are kept, and shown.
            Assert.Equal("2x_MangaJaNai_1600p_V1_ESRGAN_90k.onnx", chains[0].ModelFilePath);
            Assert.Equal(V2Onnx, chains[1].ModelFilePath);
            Assert.Equal(V2Onnx, ShownModel(pickers[1]));

            // The device list arrives: the variant rule runs.
            vm.UpscaleBackend = UpscaleBackends.Auto;
            vm.UseDeviceList(UpscaleBackends.CliTensorRtRtx, [Rtx5090]);
            Dispatcher.UIThread.RunJobs();

            Assert.Equal(MangaFp16, chains[0].ModelFilePath);
            Assert.Equal(MangaFp16, ShownModel(pickers[0]));
            Assert.Equal(V2Onnx, chains[1].ModelFilePath);
            Assert.Equal(new ModelChoice(V2Onnx, true), pickers[1].SelectedItem);
            Assert.Contains($"Chain 2's model {V2Onnx} isn't installed.", vm.ValidationText);
            Assert.False(vm.CurrentWorkflow.Valid);
            window.Close();

            // Saved: every chain keeps its model.
            var saved = JObject.Parse(JsonConvert.SerializeObject(vm, NewtonsoftJsonSuspensionDriver.Settings));
            var workflows = ((JArray)saved["Workflows"]!["$values"]!).ToList();
            var models = workflows.Select(w => ((JArray)w["Chains"]!["$values"]!).Select(c => (string)c["ModelFilePath"]!).ToList()).ToList();

            Assert.Equal([MangaFp16, V2Onnx], models[Shown]);
            foreach (var (workflow, i) in models.Select((m, i) => (m, i)).Where(x => x.i != Shown))
            {
                Assert.All(workflow, m => Assert.True(m == HatBf16 || (m.Contains("_MangaJaNai_") && m.EndsWith("_fp16.onnx")), $"workflow {i}: {m}"));
            }
            Assert.DoesNotContain(models.SelectMany(m => m), string.IsNullOrEmpty);
        }
    }
}

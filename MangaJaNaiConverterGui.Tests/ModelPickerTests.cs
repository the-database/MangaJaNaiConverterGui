using Avalonia.Collections;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using MangaJaNaiConverterGui.Services;
using MangaJaNaiConverterGui.ViewModels;
using System.Linq;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    // The chain's model ComboBox, bound as the chain template binds it: inside a DataTemplate the
    // DataContext arrives after the bindings, and the ComboBox then writes null back for a model its
    // list doesn't hold.
    public class ModelPickerTests
    {
        static readonly string[] Installed = ["a_fp16.onnx", "b_bf16.onnx", "b_fp32.onnx", "No Model"];

        static (ComboBox Picker, UpscaleChain Chain, AvaloniaList<string> Models) Picker(string model, string[]? installed = null)
        {
            var backend = new FakeBackendService(installed ?? Installed);
            var chain = new UpscaleChain(backend) { ModelFilePath = model };
            var picker = Bind(new ComboBox());
            new Window { Content = picker }.Show();
            picker.DataContext = chain;
            return (picker, chain, backend.AllModels);
        }

        public static ComboBox Bind(ComboBox picker)
        {
            picker.Bind(ItemsControl.ItemsSourceProperty, new Binding(nameof(UpscaleChain.ModelChoices)));
            picker.SelectedValueBinding = new Binding(nameof(ModelChoice.Name));
            picker.Bind(ComboBox.SelectedValueProperty, new Binding(nameof(UpscaleChain.ModelFilePath)) { Mode = BindingMode.TwoWay });
            return picker;
        }

        static string? Shown(ComboBox picker) => (picker.SelectedItem as ModelChoice)?.Name;

        [AvaloniaFact]
        public void AnInstalledModelIsShown()
        {
            var (picker, chain, _) = Picker("b_fp32.onnx");
            Assert.Equal("b_fp32.onnx", Shown(picker));
            Assert.Equal(Installed, chain.ModelChoices.Select(c => c.Name));
        }

        // A saved model that isn't installed stays saved, and the picker shows it as missing.
        [AvaloniaFact]
        public void AMissingModelKeepsItsNameAndShowsAsMissing()
        {
            var (picker, chain, _) = Picker("c_fp16.onnx");
            Assert.Equal("c_fp16.onnx", chain.ModelFilePath);
            Assert.Equal(new ModelChoice("c_fp16.onnx", true), picker.SelectedItem);
            Assert.Equal(new ModelChoice("c_fp16.onnx", true), chain.ModelChoices[^1]);
            Assert.False(chain.MissingModel);
        }

        // The picker clearing its selection never clears the model.
        [AvaloniaFact]
        public void ANullFromThePickerIsIgnored()
        {
            var (picker, chain, _) = Picker("a_fp16.onnx");
            picker.SelectedItem = null;
            Assert.Equal("a_fp16.onnx", chain.ModelFilePath);
        }

        // Rebuilding the model list (a component installed) keeps the model and its selection.
        [AvaloniaFact]
        public void AModelStaysSelectedWhenTheListIsRebuilt()
        {
            var (picker, chain, models) = Picker("b_fp32.onnx");
            var all = models.ToArray();
            models.Clear();
            models.AddRange(all);
            Assert.Equal("b_fp32.onnx", chain.ModelFilePath);
            Assert.Equal("b_fp32.onnx", Shown(picker));
        }

        // Installed later: the missing entry goes, the installed one is selected.
        [AvaloniaFact]
        public void AModelInstalledLaterIsSelected()
        {
            var (picker, chain, models) = Picker("c_fp16.onnx");
            models.Insert(0, "c_fp16.onnx");
            Dispatcher.UIThread.RunJobs();
            Assert.Equal("c_fp16.onnx", chain.ModelFilePath);
            Assert.Equal(new ModelChoice("c_fp16.onnx", false), picker.SelectedItem);
            Assert.DoesNotContain(chain.ModelChoices, c => c.Missing);
        }

        // The variant rule moving a chain, from a missing name or an installed one, shows in its picker.
        [AvaloniaFact]
        public void AVariantMoveShowsInThePicker()
        {
            var (picker, chain, _) = Picker("b_bf16.onnx");
            chain.ApplyVariant(new ModelVariants.Choice("b_fp32.onnx", null));
            Assert.Equal("b_fp32.onnx", Shown(picker));

            var (picker2, chain2, _) = Picker("a.onnx");
            chain2.ApplyVariant(new ModelVariants.Choice("a_fp16.onnx", null));
            Dispatcher.UIThread.RunJobs();
            Assert.Equal("a_fp16.onnx", Shown(picker2));
            Assert.DoesNotContain(chain2.ModelChoices, c => c.Missing);
        }

        // Picking from the list sets the model; a missing entry left behind goes.
        [AvaloniaFact]
        public void PickingAModelSetsIt()
        {
            var (picker, chain, _) = Picker("c_fp16.onnx");
            picker.SelectedItem = chain.ModelChoices.First(c => c.Name == "a_fp16.onnx");
            Dispatcher.UIThread.RunJobs();
            Assert.Equal("a_fp16.onnx", chain.ModelFilePath);
            Assert.DoesNotContain(chain.ModelChoices, c => c.Missing);
            Assert.Equal("a_fp16.onnx", Shown(picker));

            picker.SelectedItem = chain.ModelChoices.First(c => c.Name == "No Model");
            Assert.Equal("No Model", chain.ModelFilePath);
        }

        [Fact]
        public void ModelChoices()
        {
            Assert.Equal([new ModelChoice("a.onnx", false), new ModelChoice("x.onnx", true)], UpscaleChain.ModelChoicesFor(["a.onnx"], "x.onnx"));
            Assert.Equal([new ModelChoice("a.onnx", false)], UpscaleChain.ModelChoicesFor(["a.onnx"], "a.onnx"));
            Assert.Equal([new ModelChoice("a.onnx", false)], UpscaleChain.ModelChoicesFor(["a.onnx"], ""));
        }
    }
}

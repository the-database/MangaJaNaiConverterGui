using Avalonia.Collections;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Headless.XUnit;
using MangaJaNaiConverterGui.ViewModels;
using System.Linq;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    // The chain's model ComboBox, as the chain template binds it.
    public class ModelPickerTests
    {
        static (ComboBox Picker, UpscaleChain Chain, AvaloniaList<string> Models) Picker(string model)
        {
            var models = new AvaloniaList<string> { "a_fp16.onnx", "b_bf16.onnx", "b_fp32.onnx", "No Model" };
            var chain = new UpscaleChain { ModelFilePath = model };
            var picker = new ComboBox { DataContext = chain, ItemsSource = models };
            picker.Bind(ComboBox.SelectedValueProperty, new Binding(nameof(UpscaleChain.ModelFilePath)) { Mode = BindingMode.TwoWay });
            new Window { Content = picker }.Show();
            return (picker, chain, models);
        }

        // Rebuilding the model list (a component installed) clears the selection but keeps the
        // setting; ReselectModel shows it again.
        [AvaloniaFact]
        public void AModelIsReselectedAfterTheListIsRebuilt()
        {
            var (picker, chain, models) = Picker("b_fp32.onnx");
            Assert.Equal("b_fp32.onnx", picker.SelectedItem);

            var all = models.ToArray();
            models.Clear();
            models.AddRange(all);
            Assert.Equal("b_fp32.onnx", chain.ModelFilePath);

            chain.ReselectModel();
            Assert.Equal("b_fp32.onnx", picker.SelectedItem);
            Assert.Equal("b_fp32.onnx", chain.ModelFilePath);
        }

        // The variant rule moving a chain shows in its picker.
        [AvaloniaFact]
        public void AVariantMoveShowsInThePicker()
        {
            var (picker, chain, _) = Picker("b_bf16.onnx");
            chain.ApplyVariant(new Services.ModelVariants.Choice("b_fp32.onnx", null));
            Assert.Equal("b_fp32.onnx", picker.SelectedItem);
        }

        // A saved model that isn't installed stays saved (the rule may still find its variant).
        [AvaloniaFact]
        public void AMissingModelKeepsItsName()
        {
            var (picker, chain, _) = Picker("c_fp16.onnx");
            Assert.Null(picker.SelectedItem);
            Assert.Equal("c_fp16.onnx", chain.ModelFilePath);
        }
    }
}

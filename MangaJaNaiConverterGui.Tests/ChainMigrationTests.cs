using MangaJaNaiConverterGui.ViewModels;
using Xunit;

namespace MangaJaNaiConverterGui.Tests
{
    // 2.0 runs ONNX models: chains saved by 1.x move to the ONNX exports of the default models.
    public class ChainMigrationTests
    {
        [Theory]
        [InlineData("2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth", "2x_MangaJaNai_1200p_V1_ESRGAN_70k.onnx")]
        [InlineData("4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors", "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx")]
        [InlineData("2x_MangaJaNai_1200p_V1_ESRGAN_70k.onnx", "2x_MangaJaNai_1200p_V1_ESRGAN_70k.onnx")]
        // A user's own PyTorch model has no ONNX export to move to: validation asks for a conversion.
        [InlineData("4x_MyModel.pth", "4x_MyModel.pth")]
        [InlineData("No Model", "No Model")]
        [InlineData("", "")]
        public void ModelNames(string saved, string migrated) =>
            Assert.Equal(migrated, UpscaleChain.MigrateModelName(saved));

        [Theory]
        [InlineData("Auto (Estimate)", "Auto (Estimate)")]
        [InlineData("512", "512")]
        [InlineData("128", "256")]
        [InlineData("Maximum", "1024")]
        [InlineData("No Tiling", "1024")]
        [InlineData("4096", "1024")]
        [InlineData("garbage", "Auto (Estimate)")]
        public void TileSizes(string saved, string migrated) =>
            Assert.Equal(migrated, UpscaleChain.MigrateTileSize(saved));
    }
}

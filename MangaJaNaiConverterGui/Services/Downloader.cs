using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using static MangaJaNaiConverterGui.Services.Downloader;
using ProgressItem = System.Collections.Generic.KeyValuePair<long, float>;


namespace MangaJaNaiConverterGui.Services
{
    public class Downloader
    {
        public delegate void ProgressChanged(double percentage);

        public static async Task DownloadFileAsync(string url, string destinationFilePath, ProgressChanged progressChanged)
        {
            using (HttpClient client = new HttpClient())
            using (HttpResponseMessage response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead))
            {
                response.EnsureSuccessStatusCode();

                long totalBytes = response.Content.Headers.ContentLength ?? -1L;
                using (Stream contentStream = await response.Content.ReadAsStreamAsync(),
                              fileStream = new FileStream(destinationFilePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true))
                {
                    var totalRead = 0L;
                    var buffer = new byte[8192];
                    int read;

                    while ((read = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                    {
                        await fileStream.WriteAsync(buffer, 0, read);
                        totalRead += read;

                        if (totalBytes != -1)
                        {
                            double percentage = Math.Round((double)totalRead / totalBytes * 100, 0);
                            progressChanged?.Invoke(percentage);
                        }
                    }
                }
            }

            Console.WriteLine();
        }
    }

}

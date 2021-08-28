using ML.NET_BinClassTest.library.Common;
using ML.NET_BinClassTest.library.Data;
using ML.NET_BinClassTest.library.Helpers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NET_BinClassTest.library.ML
{
    public class FileClassificationFeatureExtractor
    {
        private void ExtractFolder(string folderPath, string outputFile)
        {
            if (!Directory.Exists(folderPath))
            {
                Console.WriteLine($"{folderPath} does not exist");

                return;
            }

            var files = Directory.GetFiles(folderPath);

            using (var streamWriter = new StreamWriter(Path.Combine(AppContext.BaseDirectory, $"../../../../{outputFile}")))
            {
                foreach (var file in files)
                {
                    var extractedData = new FileClassificationResponseItem(File.ReadAllBytes(file)).ToFileData();

                    extractedData.Label = !file.Contains("clean");

                    streamWriter.WriteLine(extractedData.ToString());
                }
            }

            Console.WriteLine($"Extracted {files.Length} to {outputFile}");
        }

        public void Extract(string testPath, string trainingPath)
        {
            ExtractFolder(trainingPath, Constants.SAMPLE_DATA);
            ExtractFolder(testPath, Constants.TEST_DATA);
        }
    }
}

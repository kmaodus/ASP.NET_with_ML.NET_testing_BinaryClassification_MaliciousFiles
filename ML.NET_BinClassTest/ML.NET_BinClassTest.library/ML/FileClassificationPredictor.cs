using System.IO;
using Microsoft.ML;

using ML.NET_BinClassTest.library.Data;
using ML.NET_BinClassTest.library.Helpers;
using ML.NET_BinClassTest.library.ML.Base;
using ML.NET_BinClassTest.library.ML.Objects;

namespace ML.NET_BinClassTest.library.ML
{
    public class FileClassificationPredictor : BaseML
    {
        public FileClassificationResponseItem Predict(string fileName)
        {
            var bytes = File.ReadAllBytes(fileName);
            return Predict(new FileClassificationResponseItem(bytes));
        }

        public FileClassificationResponseItem Predict(FileClassificationResponseItem file)
        {
            if (!File.Exists(Common.Constants.MODEL_PATH))
            {
                file.ErrorMessage = $"Model not found ({Common.Constants.MODEL_PATH}) - please train the model first";

                return file;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(Common.Constants.MODEL_PATH, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = MlContext.Model.Load(stream, out _);
            }

            var predictionEngine = MlContext.Model.CreatePredictionEngine<FileData, FileDataPrediction>(mlModel);

            var prediction = predictionEngine.Predict(file.ToFileData());

            file.Confidence = prediction.Probability;
            file.IsMalicious = prediction.PredictedLabel;

            return file;
        }
    }
}

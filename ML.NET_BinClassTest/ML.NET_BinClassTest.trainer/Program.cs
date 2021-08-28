using System;
using System.Text;

using ML.NET_BinClassTest.library.ML;
using ML.NET_BinClassTest.trainer.Enums;
using ML.NET_BinClassTest.trainer.Helpers;
using ML.NET_BinClassTest.trainer.Objects;

namespace ML.NET_BinClassTest.trainer
{
    class Program
    {
        public static void Main(string[] args)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            Console.Clear();

            var arguments = CommandLineParser.ParseArguments<ProgramArguments>(args);

            switch (arguments.Action)
            {
                case ProgramActions.FEATURE_EXTRACTOR:
                    new FileClassificationFeatureExtractor().Extract(arguments.TrainingFolderPath,
                        arguments.TestingFolderPath);
                    break;
                case ProgramActions.PREDICT:
                    var prediction = new FileClassificationPredictor().Predict(arguments.PredictionFileName);

                    Console.WriteLine($"File is {(prediction.IsMalicious ? "malicious" : "clean")} with a {prediction.Confidence:P2}% confidence");
                    break;
                case ProgramActions.TRAINING:
                    new FileClassificationTrainer().Train(arguments.TrainingFileName, arguments.TestingFileName);
                    break;
                default:
                    Console.WriteLine($"Unhandled action {arguments.Action}");
                    break;
            }
        }
    }
}

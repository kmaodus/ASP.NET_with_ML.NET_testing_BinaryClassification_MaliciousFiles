using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

using ML.NET_BinClassTest.library.Common;
using ML.NET_BinClassTest.library.ML.Base;
using ML.NET_BinClassTest.library.ML.Objects;

namespace ML.NET_BinClassTest.library.ML
{
    public class FileClassificationTrainer : BaseML
    {
        public void Train(string trainingFileName, string testingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName}");

                return;
            }

            if (!System.IO.File.Exists(testingFileName))
            {
                Console.WriteLine($"Failed to find test data file ({testingFileName}");

                return;
            }

            var dataView = MlContext.Data.LoadFromTextFile<FileData>(trainingFileName, hasHeader: false);

            var dataProcessPipeline = MlContext.Transforms.NormalizeMeanVariance(nameof(FileData.FileSize))
                .Append(MlContext.Transforms.NormalizeMeanVariance(nameof(FileData.Is64Bit)))
                .Append(MlContext.Transforms.NormalizeMeanVariance(nameof(FileData.IsSigned)))
                .Append(MlContext.Transforms.NormalizeMeanVariance(nameof(FileData.NumberImportFunctions)))
                .Append(MlContext.Transforms.NormalizeMeanVariance(nameof(FileData.NumberExportFunctions)))
                .Append(MlContext.Transforms.NormalizeMeanVariance(nameof(FileData.NumberImports)))
                .Append(MlContext.Transforms.Text.FeaturizeText("FeaturizeText", nameof(FileData.Strings)))
                .Append(MlContext.Transforms.Concatenate(FEATURES, nameof(FileData.FileSize), nameof(FileData.Is64Bit),
                    nameof(FileData.IsSigned), nameof(FileData.NumberImportFunctions), nameof(FileData.NumberExportFunctions),
                    nameof(FileData.NumberImports), "FeaturizeText"));

            var trainer = MlContext.BinaryClassification.Trainers.FastTree(labelColumnName: nameof(FileData.Label),
                featureColumnName: FEATURES,
                numberOfLeaves: 2,
                numberOfTrees: 1000,
                minimumExampleCountPerLeaf: 1,
                learningRate: 0.2);

            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(dataView);

            MlContext.Model.Save(trainedModel, dataView.Schema, Constants.MODEL_PATH);

            var testingDataView = MlContext.Data.LoadFromTextFile<FileData>(testingFileName, hasHeader: false);

            IDataView testDataView = trainedModel.Transform(testingDataView);

            var modelMetrics = MlContext.BinaryClassification.Evaluate(
                data: testDataView,
                labelColumnName: nameof(FileDataPrediction.Label),
                scoreColumnName: nameof(FileDataPrediction.Score));

            Console.WriteLine($"Entropy: {modelMetrics.Entropy}");
            Console.WriteLine($"Log Loss: {modelMetrics.LogLoss}");
            Console.WriteLine($"Log Loss Reduction: {modelMetrics.LogLossReduction}");
        }
    }
}

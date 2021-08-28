using ML.NET_BinClassTest.trainer.Enums;
using Constants = ML.NET_BinClassTest.library.Common.Constants;

namespace ML.NET_BinClassTest.trainer.Objects
{
    public class ProgramArguments
    {
        public ProgramActions Action { get; set; }

        public string TrainingFileName { get; set; }

        public string TestingFileName { get; set; }

        public string PredictionFileName { get; set; }

        public string ModelFileName { get; set; }

        public string TestingFolderPath { get; set; }

        public string TrainingFolderPath { get; set; }

        public ProgramArguments()
        {
            ModelFileName = Constants.MODEL_PATH;

            PredictionFileName = @"..\..\..\..\Data\predict.csv";

            TrainingFileName = $@"..\..\..\..\Data\{Constants.SAMPLE_DATA}";

            TestingFileName = $@"..\..\..\..\Data\{Constants.TEST_DATA}";
        }
    }
}
using AnomalyDetection.Commun.DataModels;
using AnomalyDetection.Commun;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using System;
using System.IO;
using System.Linq;

namespace AnomalyDetection.Predictor
{
    public class ModelsEvaluator
    {
        private readonly string _path;

        public ModelsEvaluator(string path) {
            _path = path ?? throw new ArgumentNullException(nameof(path));
        }

        public void EvaluateAllModels(int? seed = 1) {

            var env = new LocalEnvironment(seed);

            //DirectoryInfo directoryInfo = new DirectoryInfo($@"{_path}/Models/");
            //FileInfo[] filesInfo = directoryInfo.GetFiles("*.zip");

            var reader = new TextLoader(env, new TextLoader.Arguments
            {
                Column = new[] {
                    // A boolean column depicting the 'label'.
                    new TextLoader.Column("Label", DataKind.BL, 30),
                    new TextLoader.Column("V1", DataKind.R4, 1 ),
                    new TextLoader.Column("V2", DataKind.R4, 2 ),
                    new TextLoader.Column("V3", DataKind.R4, 3 ),
                    new TextLoader.Column("V4", DataKind.R4, 4 ),
                    new TextLoader.Column("V5", DataKind.R4, 5 ),
                    new TextLoader.Column("V6", DataKind.R4, 6 ),
                    new TextLoader.Column("V7", DataKind.R4, 7 ),
                    new TextLoader.Column("V8", DataKind.R4, 8 ),
                    new TextLoader.Column("V9", DataKind.R4, 9 ),
                    new TextLoader.Column("V10", DataKind.R4, 10 ),
                    new TextLoader.Column("V11", DataKind.R4, 11 ),
                    new TextLoader.Column("V12", DataKind.R4, 12 ),
                    new TextLoader.Column("V13", DataKind.R4, 13 ),
                    new TextLoader.Column("V14", DataKind.R4, 14 ),
                    new TextLoader.Column("V15", DataKind.R4, 15 ),
                    new TextLoader.Column("V16", DataKind.R4, 16 ),
                    new TextLoader.Column("V17", DataKind.R4, 17 ),
                    new TextLoader.Column("V18", DataKind.R4, 18 ),
                    new TextLoader.Column("V19", DataKind.R4, 19 ),
                    new TextLoader.Column("V20", DataKind.R4, 20 ),
                    new TextLoader.Column("V21", DataKind.R4, 21 ),
                    new TextLoader.Column("V22", DataKind.R4, 22 ),
                    new TextLoader.Column("V23", DataKind.R4, 23 ),
                    new TextLoader.Column("V24", DataKind.R4, 24 ),
                    new TextLoader.Column("V25", DataKind.R4, 25 ),
                    new TextLoader.Column("V26", DataKind.R4, 26 ),
                    new TextLoader.Column("V27", DataKind.R4, 27 ),
                    new TextLoader.Column("V28", DataKind.R4, 28 ),
                    new TextLoader.Column("Amount", DataKind.R4, 29 ),
                    //new TextLoader.Column("Features", DataKind.R4, new[] {new TextLoader.Range(1, 29) })
                },
                // First line of the file is a header, not a data row.
                HasHeader = true,
                Separator = ","
            });

            IDataView dataTest = reader.Read(new MultiFileSource($"{_path}SplitData/trainData.data"));
            ConsoleHelpers.InspectData(env, dataTest);
            //foreach (var fileInfo in filesInfo)
            //{

            ConsoleHelpers.ConsoleWriteHeader($"Predictions for saved model:");
                ITransformer model = env.ReadModel($@"{_path}/Models/cv-fastTree.zip");
                var predictionFunc = model.MakePredictionFunction<TransactionVectorModel, TransactionEstimatorModel>(env);
                ConsoleHelpers.ConsoleWriterSection($"Evaluate Data (should be predicted true):");
                dataTest.AsEnumerable<TransactionVectorModel>(env, reuseRowObject: false)
                        .Where(x => x.Label == true)
                        .Take(4)
                        .Select(testData => testData)
                        .ToList()
                        .ForEach(testData => {
                            predictionFunc.Predict(testData).PrintToConsole();
                        });


                ConsoleHelpers.ConsoleWriterSection($"Evaluate Data (should be predicted false):");
                dataTest.AsEnumerable<TransactionVectorModel>(env, reuseRowObject: false)
                        .Where(x => x.Label == false)
                        .Take(4)
                        .ToList()
                        .ForEach(testData => {
                            predictionFunc.Predict(testData).PrintToConsole();
                        });
                
            //}


        }

       
    }
}

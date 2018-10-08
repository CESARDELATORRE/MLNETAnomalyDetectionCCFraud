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

            DirectoryInfo directoryInfo = new DirectoryInfo($@"{_path}/Models/");
            FileInfo[] filesInfo = directoryInfo.GetFiles("*.ML"); 

            IDataView dataTest = new RoleMappedData(new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource($"{_path}SplitData/testData.data")),
                                    label: "Label",
                                    feature: "Features").Data;

            foreach (var fileInfo in filesInfo)
            {

                ConsoleHelpers.ConsoleWriteHeader($"Predictions for Model: {fileInfo.Name}:");

                ITransformer model = env.ReadModel(fileInfo.FullName);
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
                
            }


        }

       
    }
}

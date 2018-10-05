
using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using AnomalyDetection.Train.DataModels;
using AnomalyDetection.Train.Helpers;
using System;

namespace AnomalyDetection.Train
{
    class Program
    {
        static void Main(string[] args)
        {
            var Path = @"Data/";
            var dataPath = $"Data/creditcard.csv";


            if (!File.Exists(dataPath)) {
                ZipFile.ExtractToDirectory($"../../../Data/creditcardfraud.zip", Path);
            }


            // Create a new environment for ML.NET operations.
            // It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            // Seed set to any number so you have a deterministic environment
            var env = new LocalEnvironment(seed: 1);

            // Step one: read the data as an IDataView.

            // Create the reader: define the data columns 
            // and where to find them in the text file.
            var reader = TextLoader.CreateReader(env,
                            ctx => (
                                // V1...V28 + Amount
                                Features: ctx.LoadFloat(1, 29),
                                // Class
                                Label: ctx.LoadBool(30)),
                                separator: ',', hasHeader: true);

            // Now read the file 
            // (remember though, readers are lazy, so the actual 
            //  reading will happen when the data is accessed).
            var data = reader.Read(new MultiFileSource(dataPath));

            // lets inspect data
            ConsoleHelpers.ConsoleWriteHeader("Show 4");
            data.AsDynamic
                // Convert to an enumerable of user-defined type. 
                .AsEnumerable<TransactionVectorModel>(env, reuseRowObject: false)
                .Where(x=>x.Label == true)
                // Take a couple values as an array.
                .Take(2)
                .ToList()
                // print to console
                .ForEach(row => { row.PrintToConsole(); });

            data.AsDynamic
                // Convert to an enumerable of user-defined type. 
                .AsEnumerable<TransactionVectorModel>(env, reuseRowObject: false)
                .Where(x => x.Label == false)
                // Take a couple values as an array.
                .Take(2)
                .ToList()
                // print to console
                .ForEach(row => { row.PrintToConsole(); });

            // Step two: define the learning pipeline. 

            // We know that this is a regression task, so we create a regression context: it will give us the algorithms
            // we need, as well as the evaluation procedure.
            var classification = new BinaryClassificationContext(env);

            var learningPipeline = reader.MakeNewEstimator()
                   // normalize values
                   .Append(row => (
                            FeaturesNormalizedByMeanVar: row.Features.NormalizeByMeanVar(),
                            row.Label))
                   .Append(row => (
                            row.Label,
                            Predictions: classification.Trainers.Sdca( row.Label, row.FeaturesNormalizedByMeanVar)
                   )
                );

            // Step three: Train the model.
            // Split the data 80:20 into train and test sets, train and evaluate.
            var (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);
            //var (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2, stratificationColumn: row => row.Label);


            var model = learningPipeline.Fit(trainData);

            // Compute quality metrics on the test set.

            ConsoleHelpers.ConsoleWriteHeader("Train Metrics (80/20) :");
            var metrics = classification.Evaluate(model.Transform(testData), row => row.Label, row => row.Predictions);
            metrics.ToConsole();
            

            // Now run the n-fold cross-validation experiment, using the same pipeline.
            int numFolds = 5;
            var cvResults = classification.CrossValidate(trainData, learningPipeline, r => r.Label, numFolds: numFolds);

            // Let's get Cross Validate metrics           
            int count = 1;
            var cvModels = cvResults.ToList();

            cvModels.ForEach(result =>
            {
                ConsoleHelpers.ConsoleWriteHeader($"Train Metrics Cross Validate [{count++}/{numFolds}]:");
                classification.Evaluate(result.model.Transform(testData), row => row.Label, row => row.Predictions);
                metrics.ToConsole();
            });
            
            ConsoleHelpers.ConsolePressAnyKey();
        }
    }

}

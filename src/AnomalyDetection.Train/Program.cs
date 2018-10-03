using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;

namespace AnomalyDetection.Train
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataPath = @"Data/creditcard.csv";

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
                    Features: ctx.LoadFloat(1,29), // V1...V28 + Amount
                    Label: ctx.LoadText(30)), // Class
                    separator: ',', hasHeader: true);

            // Now read the file 
            // (remember though, readers are lazy, so the actual 
            //  reading will happen when the data is accessed).
            var data = reader.Read(new MultiFileSource(dataPath));

            // 'transformedData' is a 'promise' of data. Let's actually read it.
            var someRows = data.AsDynamic
                // Convert to an enumerable of user-defined type. 
                .AsEnumerable<TransactionData>(env, reuseRowObject: false)
                // Take a couple values as an array.
                .Take(4).ToArray();

            ConsoleHelpers.ConsoleWriteHeader("Show 4");
            foreach(var viewRow in someRows)
            {            
                Console.WriteLine($"Label: {viewRow.Label}");
                Console.WriteLine($"Features: [0] {viewRow.Features[0]} [1] {viewRow.Features[1]} [2] {viewRow.Features[2]} ... [28] {viewRow.Features[28]}");
                //Console.WriteLine($"Features Normalized: [0] {viewRow.FeaturesNormalizedByMeanVar[0]} [1] {viewRow.FeaturesNormalizedByMeanVar[1]} [2] {viewRow.FeaturesNormalizedByMeanVar[2]} ... [28] {viewRow.FeaturesNormalizedByMeanVar[28]}");
            }
            Console.WriteLine("");   
             
            // Step two: define the learning pipeline. 

            // We know that this is a regression task, so we create a regression context: it will give us the algorithms
            // we need, as well as the evaluation procedure.
            var classification = new MulticlassClassificationContext(env);
            

            // Start creating our processing pipeline. 
            var learningPipeline = reader.MakeNewEstimator()                      
                   .Append(row => ( 
                      FeaturesNormalizedByMeanVar : row.Features.NormalizeByMeanVar(), // normalize values
                      Label : row.Label.ToKey()))
                   .Append(row => ( 
                           row.Label , 
                           Predictions :  classification.Trainers.Sdca(row.Label, features: row.FeaturesNormalizedByMeanVar )));
            
            // Split the data 80:20 into train and test sets, train and evaluate.
            var (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);

            // Step three: Train the model.
            var model = learningPipeline.Fit(trainData);
            // Compute quality metrics on the test set.
            var metrics = classification.Evaluate(model.Transform(testData), row => row.Label ,row => row.Predictions);


            ConsoleHelpers.ConsoleWriteHeader("Train Metrics (80/20) :");
            Console.WriteLine($"Acuracy Macro: {metrics.AccuracyMacro}");
            Console.WriteLine($"Acuracy Micro: {metrics.AccuracyMicro}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");
            Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction}");



            // Now run the 5-fold cross-validation experiment, using the same pipeline.
            var cvResults = classification.CrossValidate(data, learningPipeline, r => r.Label, numFolds: 5);

            // The results object is an array of 5 elements. For each of the 5 folds, we have metrics, model and scored test data.
            // Let's compute the average micro-accuracy.
            
            var cvmetrics = cvResults.Select(r => r.metrics);
            int count = 1;
            foreach(var metric in cvmetrics){
                ConsoleHelpers.ConsoleWriteHeader($"Train Metrics Cross Validate [{count}/5]:");
                Console.WriteLine($"Acuracy Macro: {metrics.AccuracyMacro}");    
                Console.WriteLine($"Acuracy Micro: {metrics.AccuracyMicro}");
                Console.WriteLine($"Log Loss: {metrics.LogLoss}");
                Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction}"); 
                Console.WriteLine("");  
                count++;             
            }
        }
    }

    public class TransactionData
    {
        public string Label;
        public float[] Features;
    }

    public class TransactionNormalizedData
    {
        public string Label;
        public float[] Features;
        public float[] FeaturesNormalizedByMeanVar;
    }

    public static class ConsoleHelpers
    {
        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new String('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            Console.WriteLine(" ");
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new String('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }
    }
}

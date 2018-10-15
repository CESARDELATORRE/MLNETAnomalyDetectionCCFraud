﻿using AnomalyDetection.Common;
using AnomalyDetection.Common.DataModels;

using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.FastTree;
using static Microsoft.ML.Runtime.Api.GenerateCodeCommand;

using System;
using System.IO;
using System.Linq;

namespace AnomalyDetection.Train
{
    public class ModelBuilder
    {
        private readonly string _assetsPath;
        private readonly string _dataSetFile;
        private readonly string _outputPath;

        private BinaryClassificationContext _context;
        private TextLoader _reader;
        private IDataView _trainData;
        private IDataView _testData;
        private LocalEnvironment _env;


        // todo all with dinamic APi
        public ModelBuilder(string assetsPath, string dataSetFile)
        {
            _assetsPath = assetsPath ?? throw new ArgumentNullException(nameof(assetsPath));
            _dataSetFile = dataSetFile ?? throw new ArgumentNullException(nameof(dataSetFile));
            _outputPath = Path.Combine(_assetsPath, "output");
        }


        public ModelBuilder Build(int? seed = 1)
        {

            // Create a new environment for ML.NET operations.
            // It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            // Seed set to any number so you have a deterministic environment
            _env = new LocalEnvironment(seed);
            (_context, _reader, _trainData, _testData) = PrepareData(_env);

            return this;
        }

        public void TrainFastTreeAndSaveModels( int cvNumFolds = 2, int numLeaves= 20 , int numTrees = 100,
                                                int minDocumentsInLeafs = 10, double learningRate = 0.2,
                                                Action<Arguments> advancedSettings = null)
        {

            var logMeanVarNormalizer =   new Normalizer(_env,Normalizer.NormalizerMode.MeanVariance ,("Features", "FeaturesNormalizedByMeanVar"));

            var estimator = new ConcatEstimator(_env, "Features", new[] { "Amount", "V1", "V2", "V3", "V4", "V5", "V6",
                                                                          "V7", "V8", "V9", "V10", "V11", "V12",
                                                                          "V13", "V14", "V15", "V16", "V17", "V18",
                                                                          "V19", "V20", "V21", "V22", "V23", "V24",
                                                                          "V25", "V26", "V27", "V28" })                      
                        .Append(new Normalizer(_env, Normalizer.NormalizerMode.MeanVariance, ("Features", "FeaturesNormalizedByMeanVar")))
                        .Append(new FastTreeBinaryClassificationTrainer(_env, "Label", "Features",
                                                    
                                                    numLeaves: 20,
                                                    numTrees: 100,
                                                    minDocumentsInLeafs : 10,
                                                    learningRate: 0.2));
            

            // Now run the n-fold cross-validation experiment, using the same pipeline.
            // Can't do stratification when column type is a boolean
            // var cvResults = _context.CrossValidate(_trainData, estimator, labelColumn: "Label", numFolds: numFolds, stratificationColumn: "Label");
            var cvResults = _context.CrossValidate(_trainData, estimator, labelColumn: "Label", numFolds: cvNumFolds);


            // Let's get Cross Validate metrics           
            int count = 1;
            var cvModels = cvResults.ToList();
            cvModels.ForEach(result =>
            {
                ConsoleHelpers.ConsoleWriteHeader($"Train Metrics Cross Validate [{count}/{cvNumFolds}]:");
                result.metrics.ToConsole();
                ConsoleHelpers.ConsoleWriteHeader($"Show 4 [model {count}]");
                ConsoleHelpers.InspectScoredData(_env, result.scoredTestData);
                count++;
            });
            // save model with best accuracy
            var bestmodel = cvModels.OrderByDescending(result => result.metrics.Accuracy).Select(result => result.model).FirstOrDefault();           
            bestmodel.SaveModel(_env, Path.Combine(_outputPath, "cv-fastTree.zip"));
            Console.WriteLine("");
            Console.WriteLine("Saved best model.");

        }

        private (BinaryClassificationContext context, TextLoader, IDataView trainData, IDataView testData) 
                    PrepareData(LocalEnvironment env)
        {

            IDataView data = null;
            IDataView trainData = null;
            IDataView testData = null;

            // Step one: read the data as an IDataView.
            // Create the reader: define the data columns 
            // and where to find them in the text file.
            var reader = new TextLoader(env, new TextLoader.Arguments
            {
                Column = new[] {
                    // A boolean column depicting the 'label'.
                    new TextLoader.Column("Label", DataKind.BL, 30),
                    // 29 Features V1..V28 + Amount
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
                },
                // First line of the file is a header, not a data row.
                HasHeader = true,
                Separator = ","
            });


            // We know that this is a Binary Classification task,
            // so we create a Binary Classification context:
            // it will give us the algorithms we need,
            // as well as the evaluation procedure.
            var classification = new BinaryClassificationContext(env);

            if (!File.Exists(Path.Combine(_outputPath, "testData.idv")) &&
                !File.Exists(Path.Combine(_outputPath, "trainData.idv"))){
                // Split the data 80:20 into train and test sets, train and evaluate.

                data = reader.Read(new MultiFileSource(_dataSetFile));
                ConsoleHelpers.ConsoleWriteHeader("Show 4 (source)");
                ConsoleHelpers.InspectData(env, data);



                // Can't do stratification when column type is a boolean
                //(trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2, stratificationColumn: "Label");
                (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);

                // save test split
                using (var ch = env.Start("SaveData"))
                using (var file = env.CreateOutputFile(Path.Combine(_outputPath, "testData.idv")))
                {
                    var saver = new BinarySaver(env, new BinarySaver.Arguments());
                    DataSaverUtils.SaveDataView(ch, saver, testData, file);
                }

                // save train split
                using (var ch = env.Start("SaveData"))
                using (var file = env.CreateOutputFile(Path.Combine(_outputPath, "trainData.idv")))
                {
                    var saver = new BinarySaver(env, new BinarySaver.Arguments());
                    DataSaverUtils.SaveDataView(ch, saver, trainData, file);
                }

            }
            else {
                // Load splited data
                var binTrainData = new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource(Path.Combine(_outputPath, "trainData.idv")));
                var trainRoles = new RoleMappedData(binTrainData, roles: TransactionVectorModel.Roles());
                trainData = trainRoles.Data;


                var binTestData = new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource(Path.Combine(_outputPath, "testData.idv")));
                var testRoles = new RoleMappedData(binTestData, roles: TransactionVectorModel.Roles());
                testData = testRoles.Data;

            }

            ConsoleHelpers.ConsoleWriteHeader("Show 4 (traindata)");
            ConsoleHelpers.InspectData(env, trainData);

            ConsoleHelpers.ConsoleWriteHeader("Show 4 (testData)");
            ConsoleHelpers.InspectData(env, testData);

            return (classification, reader, trainData, testData);
        }
    }

}

using AnomalyDetection.Commun;

using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;

using System;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace AnomalyDetection.Train
{
    public class ModelBuilder
    {
        private readonly string _path;
        private const string _fileName = "creditcard.csv";
        private const string _zipFile = "../../../Data/creditcardfraud.zip";

        private BinaryClassificationContext _context;
        private TextLoader _reader;
        private IDataView _trainData;
        private IDataView _testData;
        private LocalEnvironment _env;


        // todo all with dinamic APi
        public ModelBuilder(string path)
        {
            _path = path ?? throw new ArgumentNullException(nameof(path));
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

        public void TrainSaveModels(int numFolds = 2)
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
            
            var model = estimator.Fit(_trainData);

            // Now run the n-fold cross-validation experiment, using the same pipeline.

            // Can't do stratification when column type is a boolean
            // var cvResults = _context.CrossValidate(_trainData, estimator, labelColumn: "Label", numFolds: numFolds, stratificationColumn: "Label");
            var cvResults = _context.CrossValidate(_trainData, estimator, labelColumn: "Label", numFolds: numFolds);


            // Let's get Cross Validate metrics           
            int count = 1;
            var cvModels = cvResults.ToList();
            cvModels.ForEach(result =>
            {
                ConsoleHelpers.ConsoleWriteHeader($"Train Metrics Cross Validate [{count++}/{numFolds}]:");
                result.metrics.ToConsole();
                ConsoleHelpers.InspectScoredData(_env, result.scoredTestData);

                // save ML model to disk
                //result.model.SaveModel(_env, $"{_path}Models/cv{count - 1}-fastTree.zip");
            });
            var bestmodel = cvModels.OrderByDescending(result => result.metrics.Accuracy).Select(result => result.model).FirstOrDefault();
            bestmodel.SaveModel(_env, $"{_path}Models/cv-fastTree.zip");
            //todo save with best acurracy
        }

        private (BinaryClassificationContext context,
                   TextLoader,
                   IDataView trainData,
                   IDataView testData)
                PrepareData(LocalEnvironment env)
        {
            if (!File.Exists($"{_path}{_fileName}"))
            {
                ZipFile.ExtractToDirectory(_zipFile, _path);
            }
            if (!Directory.Exists($"{_path}Models/"))
            {
                Directory.CreateDirectory($"{_path}Models/");
            }

            // Step one: read the data as an IDataView.

            // Create the reader: define the data columns 
            // and where to find them in the text file.
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
                    new TextLoader.Column("Amount", DataKind.R4, 29 )
                },
                // First line of the file is a header, not a data row.
                HasHeader = true,
                Separator = ","
            });

            // Create the reader: define the data columns and where to find them in the text file.
            //var reader = TextLoader.CreateReader(env, ctx => (
            //        // A boolean column depicting the 'target label'.
            //        V1: ctx.LoadFloat(1),
            //        V2: ctx.LoadFloat(2),
            //        V3: ctx.LoadFloat(3),
            //        V4: ctx.LoadFloat(4),
            //        V5: ctx.LoadFloat(5),
            //        V6: ctx.LoadFloat(6),
            //        V7: ctx.LoadFloat(7),
            //        V8: ctx.LoadFloat(8),
            //        V9: ctx.LoadFloat(9),
            //        V10: ctx.LoadFloat(10),
            //        V11: ctx.LoadFloat(11),
            //        V12: ctx.LoadFloat(12),
            //        V13: ctx.LoadFloat(13),
            //        V14: ctx.LoadFloat(14),
            //        V15: ctx.LoadFloat(15),
            //        V16: ctx.LoadFloat(16),
            //        V17: ctx.LoadFloat(17),
            //        V18: ctx.LoadFloat(18),
            //        V19: ctx.LoadFloat(19),
            //        V20: ctx.LoadFloat(20),
            //        V21: ctx.LoadFloat(21),
            //        V22: ctx.LoadFloat(22),
            //        V23: ctx.LoadFloat(23),
            //        V24: ctx.LoadFloat(24),
            //        V25: ctx.LoadFloat(25),
            //        V26: ctx.LoadFloat(26),
            //        V27: ctx.LoadFloat(27),
            //        V28: ctx.LoadFloat(28),
            //        // Three text columns.
            //        Amount: ctx.LoadFloat(29),
            //        Label: ctx.LoadBool(30)),
            //    hasHeader: true,
            //    separator: ',');

            // Now read the file 
            IDataView data = null;


            // We know that this is a Binary Classification task,
            // so we create a Binary Classification context:
            // it will give us the algorithms we need,
            // as well as the evaluation procedure.
            var classification = new BinaryClassificationContext(env);

            IDataView trainData = null;
            IDataView testData = null;

            // Step 2: split Data
            if (!Directory.Exists($"{_path}SplitData/"))
            {
                data = reader.Read(new MultiFileSource($"{_path}{_fileName}"));
                // (remember though, readers are lazy, so the actual 
                //  reading will happen when the data is accessed).

                ConsoleHelpers.ConsoleWriteHeader("Show 4 (source)");
                ConsoleHelpers.InspectData(env, data);

                // Split the data 80:20 into train and test sets, train and evaluate.

                // Can't do stratification when column type is a boolean
                //(trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2, stratificationColumn: "Label");
                (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);

                Directory.CreateDirectory($"{_path}SplitData/");

                // save to files
                if (!File.Exists($"{_path}SplitData/testData.data"))
                {
                    var saver = new BinarySaver(env, new BinarySaver.Arguments());
                    using (var ch = env.Start("SaveData"))
                    using (var file = env.CreateOutputFile($"{_path}SplitData/testData.data"))
                    {
                        DataSaverUtils.SaveDataView(ch, saver, testData, file);
                    }
                }
                if (!File.Exists($"{_path}SplitData/trainData.data"))
                {
                    var saver = new BinarySaver(env, new BinarySaver.Arguments());
                    using (var ch = env.Start("SaveData"))
                    using (var file = env.CreateOutputFile($"{_path}SplitData/trainData.data"))
                    {
                        DataSaverUtils.SaveDataView(ch, saver, trainData, file);
                    }
                }
            }
            else
            {
                // Load from files
                trainData = reader.Read(new MultiFileSource($"{_path}SplitData/trainData.data"));
                testData = reader.Read(new MultiFileSource($"{_path}SplitData/testData.data"));
            }

            ConsoleHelpers.ConsoleWriteHeader("Show 4 (traindata)");
            ConsoleHelpers.InspectData(env, trainData);

            ConsoleHelpers.ConsoleWriteHeader("Show 4 (testData)");
            ConsoleHelpers.InspectData(env, testData);

            return (classification, reader, trainData, testData);
        }
    }

}

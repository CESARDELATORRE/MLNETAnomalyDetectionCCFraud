using AnomalyDetection.Commun;

using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
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
        private DataReader<IMultiStreamSource, (Vector<float> Features, Scalar<bool> Label)> _reader;
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
            var estimator = _reader.MakeNewEstimator()
              // normalize values
              .Append(row => (
                       FeaturesNormalizedByMeanVar: row.Features.NormalizeByMeanVar(),
                       row.Label))
              .Append(row => (
                       row.Label,
                       Predictions: _context.Trainers.FastTree(row.Label, row.FeaturesNormalizedByMeanVar)));

            var model = estimator.AsDynamic.Fit(_trainData);

            // Now run the n-fold cross-validation experiment, using the same pipeline.
            var cvResults = _context.CrossValidate(_trainData, estimator.AsDynamic, labelColumn: "Label", numFolds: numFolds);
            // Let's get Cross Validate metrics           
            int count = 1;
            var cvModels = cvResults.ToList();
            cvModels.ForEach(result =>
            {
                ConsoleHelpers.ConsoleWriteHeader($"Train Metrics Cross Validate [{count++}/{numFolds}]:");
                result.metrics.ToConsole();
                ConsoleHelpers.InspectScoredData(_env, result.scoredTestData);

                // save ML model to disk
                result.model.SaveModel(_env, $"{_path}Models/cv{count - 1}-fastTree.zip");
            });

            // todo save with best acurracy
        }

        private (BinaryClassificationContext context,
                   DataReader<IMultiStreamSource, (Vector<float> Features, Scalar<bool> Label)> reader,
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
            var reader = TextLoader.CreateReader(env,
                            ctx => (
                                // V1...V28 + Amount
                                Features: ctx.LoadFloat(1, 29),
                                // Class
                                Label: ctx.LoadBool(30)),
                                separator: ',', hasHeader: true);

            // Now read the file 
            var data = reader.Read(new MultiFileSource($"{_path}{_fileName}")).AsDynamic;
            // (remember though, readers are lazy, so the actual 
            //  reading will happen when the data is accessed).
            ConsoleHelpers.InspectData(env, data);

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
                trainData = new RoleMappedData(new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource($"{_path}SplitData/trainData.data")),
                                               label: "Label",
                                               feature: "Features").Data;

                testData = new RoleMappedData(new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource($"{_path}SplitData/testData.data")),
                                                label: "Label",
                                                feature: "Features").Data;
            }

            return (classification, reader, trainData, testData);
        }
    }

}

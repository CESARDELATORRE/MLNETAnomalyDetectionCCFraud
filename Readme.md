# Customer Segmentation - Clustering sample

## DataSet

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Obtained from [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML

By: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

## Problem

## ML Task - [Binary Classification](https://en.wikipedia.org/wiki/Binary_classification)

Binary or binomial classification is the task of classifying the elements of a given set into two groups (predicting which group each one belongs to) on the basis of a classification rule. Contexts requiring a decision as to whether or not an item has some qualitative property, some specified characteristic
  

## Solution
tbd

### 1. Build model
tbd

`````csharp
   // Load Model
   var reader = TextLoader.CreateReader(env,
                ctx => (
                    // V1...V28 + Amount
                    Features: ctx.LoadFloat(1, 29),
                    // Class
                    Label: ctx.LoadText(30)),
                    separator: ',', hasHeader: true);
`````

### 2. Train model
tbd

`````csharp
    var classification = new BinaryClassificationContext(env);

    var estimator = reader.MakeNewEstimator()
            // normalize values
            .Append(row => (
                    FeaturesNormalizedByMeanVar: row.Features.NormalizeByMeanVar(),
                    row.Label))
            .Append(row => (
                    row.Label,
                    Predictions: classification.Trainers.FastTree( row.Label, row.FeaturesNormalizedByMeanVar)
            )
        );

    [...]

    // Split the data 80:20 into train and test sets, train and evaluate.
    var (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);
    var model = estimator.Fit(trainData);
`````

### 3. Evaluate model
tbd

`````csharp
[...]
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
        result.model.SaveModel(_env, $"{_path}Models/cv{count - 1}-fastTree.ML");
    });
[...]
`````

### 4. Consume model
tbd

`````csharp
[...]
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
[...]
`````
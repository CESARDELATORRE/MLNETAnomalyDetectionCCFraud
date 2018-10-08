using Microsoft.ML.Runtime.Api;
using System;

namespace AnomalyDetection.Commun.DataModels
{

    public interface IModelEntity {
        void PrintToConsole();
    }

    public class TransactionVectorModel : IModelEntity
    {
        public bool Label;
        [VectorType(28)]
        public float[] Features;

        public void PrintToConsole() {
            Console.WriteLine($"Label: {Label}");
            Console.WriteLine($"Features: [0] {Features[0]} [1] {Features[1]} [2] {Features[2]} ... [28] {Features[28]}");
        }
    }

    public class TransactionEstimatorModel : IModelEntity
    {
        public bool Label;
        public bool PredictedLabel;
        public float Score;
        public float Probability;

        public void PrintToConsole()
        {
            Console.WriteLine($"Predicted Label: {PredictedLabel} [{Label}]");
            Console.WriteLine($"Probability: {Probability}  ({Score})");
        }
    }
}

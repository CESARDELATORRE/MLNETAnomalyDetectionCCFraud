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
        public float V1;
        public float V2;
        public float V3;
        public float V4;
        public float V5;
        public float V6;
        public float V7;
        public float V8;
        public float V9;
        public float V10;
        public float V11;
        public float V12;
        public float V13;
        public float V14;
        public float V15;
        public float V16;
        public float V17;
        public float V18;
        public float V19;
        public float V20;
        public float V21;
        public float V22;
        public float V23;
        public float V24;
        public float V25;
        public float V26;
        public float V27;
        public float V28;
        public float Amount;
        //[VectorType(28)]
        //public float[] Features;

        public void PrintToConsole() {
            Console.WriteLine($"Label: {Label}");
            Console.WriteLine($"Features: [V1] {V1} [V2] {V2} [V3] {V3} ... [V28] {V28} Amount: {Amount}");
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

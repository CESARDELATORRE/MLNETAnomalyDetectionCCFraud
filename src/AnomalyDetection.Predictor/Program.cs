using AnomalyDetection.Commun;
using System;

namespace AnomalyDetection.Predictor
{
    class Program
    {
        static void Main(string[] args)
        {
            var path = @"../../../../../Data/";
            var modelEvaluator = new ModelsEvaluator(path);
            modelEvaluator.EvaluateAllModels();
            ConsoleHelpers.ConsolePressAnyKey();
        }
    }
}

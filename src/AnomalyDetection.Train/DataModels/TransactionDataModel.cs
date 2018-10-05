using System;

namespace AnomalyDetection.Train.DataModels
{
    public class TransactionVectorModel
    {
        public bool Label;
        public float[] Features;

        public void PrintToConsole() {
            Console.WriteLine($"Label: {Label}");
            Console.WriteLine($"Features: [0] {Features[0]} [1] {Features[1]} [2] {Features[2]} ... [28] {Features[28]}");
        }
    }

}

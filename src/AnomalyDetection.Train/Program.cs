using AnomalyDetection.Commun;

namespace AnomalyDetection.Train
{
    class Program
    {
        static void Main(string[] args)
        {
            var path =          @"../../../../../Data/";
            var modelBuilder = new ModelBuilder(path);
            modelBuilder.Build();
            modelBuilder.TrainSaveModels();
            ConsoleHelpers.ConsolePressAnyKey();
        }
    } 
}

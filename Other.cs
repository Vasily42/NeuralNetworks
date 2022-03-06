namespace NeuralNetwork;

internal static class StGeneral
{
    static public float NextFloat() =>
    (float)s_stRandom.NextDouble();
    private static readonly Random s_stRandom;

    static StGeneral()
    {
        s_stRandom = new Random();
    }
}



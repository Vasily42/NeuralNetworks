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

public struct Parameter
{
    public Parameter(float parameterInitValue)
    {
        value = parameterInitValue;
        gradient = 0;
        firstMomentum = 0;
        secondMomentum = 0;
    }

    public float value;
    public float gradient;
    public float firstMomentum;
    public float secondMomentum;
}

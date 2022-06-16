namespace NeuralNetwork;

interface IParameterized
{
    void Correction();
    void Reset();
}

[Serializable]
public class Flatten : Layer
{
    public Flatten(string name = null) : base(name)
    {
    }

    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = new(inputShape.n0, inputShape.nF1);
    }

    public sealed override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        nextLayer.Forward(input, in actualMBSize, in training);
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize)
    {
        prevLayer.BackProp(deriv, in actualMBSize);
    }
}

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

public class LayerCommander 
{
    private Layer[] layers;

    public void AddLayer(Layer layer)
    {
        if (layers == null)
        {
            layers = new Layer[1]{layer};
            return;
        }

        Layer[] newLayers = new Layer[layers.Length + 1];
        for (int i = 0; i < layers.Length; i++) newLayers[i] = layers[i];
        newLayers[^1] = layer;
        layers = newLayers;
    }

    public void ResetForAll() { for (int i = 0; i < layers.Length; i++) layers[i].Reset(); }
    public void CorrectionForAll() { for (int i = 0; i < layers.Length; i++) layers[i].ParameterCorrection(); }
}


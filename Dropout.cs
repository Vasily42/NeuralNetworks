namespace NeuralNetwork;

public unsafe class Dropout : Layer
{
    private readonly ushort dropoutRate;
    private readonly float dropoutRateFloat;
    private readonly float qCoeff, invDropoutRateFloat;

    private readonly Random rndDropout;
    private bool[] dropped;

    public Dropout(float dropoutRate)
    {
        this.dropoutRateFloat = (float)Math.Round(dropoutRate, 3);
        this.dropoutRate = (ushort)(1000 * dropoutRateFloat);
        this.invDropoutRateFloat = (1 - dropoutRateFloat);
        this.qCoeff = 1 / invDropoutRateFloat;
        rndDropout = new Random();
    }

    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = inputShape;

        input = Tensor.Create(inputShape);

        outputDerivatives = Tensor.Create(outputShape);

        dropped = new bool[inputShape.flatBatchSize];
    }

    public sealed override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        input.CopyTo(this.input);

        if (!training)
        {
            for (int batch = 0; batch < actualMBSize; batch++)
            {
                for (int flat = 0; flat < inputShape.flatBatchSize; flat++)
                    this.input[batch, flat] *= invDropoutRateFloat;
            }
        }
        else
        {
            for (int i = 0; i < dropped.Length; i++)
                dropped[i] = rndDropout.Next(1000) < dropoutRate;

            for (int batch = 0; batch < actualMBSize; batch++)
            {
                Drop(this.input, in batch);
            }
        }

        nextLayer.Forward(this.input, in actualMBSize, in training);
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize)
    {
        deriv.CopyTo(outputDerivatives);

        for (int batch = 0; batch < actualMBSize; batch++)
        {
            Drop(outputDerivatives, in batch);
        }

        prevLayer.BackProp(outputDerivatives, in actualMBSize);
    }

    public void Drop(Tensor input, in int batch)
    {
        for (int i = 0; i < inputShape.flatBatchSize; i++)
            if (dropped[i]) input[batch, i] = 0;
            else input[batch, i] *= qCoeff;
    }
}

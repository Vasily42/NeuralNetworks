namespace NeuralNetwork;

public unsafe abstract class Loss
{
    protected float lossValue;
    public float LossValue => lossValue;

    public static Loss CreateLoss(string lossName) => lossName.ToLower() switch
    {
        "mae" => new MeanAbsoluteError(),
        "mse" => new MeanSquaredError(),
        "rms" => new RootMeanSquaredError(),
        "cross entropy" or "ce" => new CrossEntropy(),
        _ => throw new Exception(),
    };

    public virtual Tensor Derivate(Tensor ideal, Tensor predicted)
    {
        Tensor derivatives = new(predicted.shape);

        lossValue = 0;

        void derivationAction(int batch)
        {
            for (int i = 0; i < predicted.shape.flatBatchSize; i++)
            {
                derivatives[batch, i] = PartialDerivative(ideal[batch, i], predicted[batch, i]);
                lossValue += PartialError(ideal[batch, i], predicted[batch, i]);
            }
        }

        for (int i = 0; i < ideal.shape.batchSize; i++)
            derivationAction(i);

        lossValue = LossGen(lossValue, ideal.shape.batchSize);
        lossValue /= ideal.shape.batchSize;

        return derivatives;
    }

    public virtual float ComputeError(Tensor ideal, Tensor predicted)
    {
        lossValue = 0;

        void errCalculatingAction(int batch)
        {
            for (int i = 0; i < ideal.shape.flatBatchSize; i++)
                lossValue += PartialError(ideal[batch, i], predicted[batch, i]);
        }

        for (int i = 0; i < ideal.shape.batchSize; i++)
            errCalculatingAction(i);

        lossValue = LossGen(lossValue, ideal.shape.batchSize);
        return lossValue;
    }

    protected virtual float PartialError(float ideal, float pred) => 0;
    protected virtual float PartialDerivative(float ideal, float pred) => 0;
    protected virtual float LossGen(float loss, float batchSize) => 0;
}

public unsafe class MeanSquaredError : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => MathF.Pow(ideal - pred, 2);
    protected sealed override float PartialDerivative(float ideal, float pred) => pred - ideal;
    protected sealed override float LossGen(float sum, float batchSize) => sum / batchSize;
}

public unsafe class MeanAbsoluteError : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => MathF.Abs(ideal - pred);
    protected sealed override float PartialDerivative(float ideal, float pred) => -MathF.Sign(ideal - pred);
    protected sealed override float LossGen(float sum, float batchSize) => sum / batchSize;
}

public unsafe class RootMeanSquaredError : Loss
{
    public sealed override Tensor Derivate(Tensor ideal, Tensor predicted)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.batchSize;

        float avCoeff = 1 / ideal.shape.flatBatchSize;

        Tensor derivatives = new(predicted.shape);

        float sum;

        for (int i = 0, j; i < actualMBSize; i++)
        {
            sum = 0;
            for (j = 0; j < predicted.shape.flatBatchSize; j++)
                sum += PartialError(ideal[i, j], predicted[i, j]);

            sum = MathF.Sqrt(avCoeff * sum);

            for (j = 0; j < predicted.shape.flatBatchSize; j++)
                derivatives[i, j] = -avCoeff * (ideal[i, j] - predicted[i, j]) / (sum + 1.0E-8F);

            lossValue += sum;
        }

        lossValue /= actualMBSize;

        return derivatives;
    }

    public sealed override float ComputeError(Tensor ideal, Tensor predicted)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.batchSize;

        float avCoeff = 1 / ideal.shape.flatBatchSize;

        float sum;

        for (int i = 0, j; i < actualMBSize; i++)
        {
            sum = 0;
            for (j = 0; j < predicted.shape.flatBatchSize; j++)
                sum += PartialError(ideal[i, j], predicted[i, j]);

            lossValue += MathF.Sqrt(avCoeff * sum);
        }

        lossValue /= actualMBSize;

        return lossValue;
    }

    protected sealed override float PartialError(float ideal, float pred) => MathF.Pow(ideal - pred, 2);
}



public unsafe class CrossEntropy : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => ideal * MathF.Log(pred + 1.0E-8F);
    protected sealed override float PartialDerivative(float ideal, float pred) => -ideal / (pred + 1.0E-8F);
    protected sealed override float LossGen(float sum, float batchSize) => -sum;
}

namespace NeuralNetwork;

[Serializable]
public unsafe abstract class Loss
{
    protected float lossValue;
    public float LossValue => lossValue;

    public static Loss GetLoss(string lossName)
    {
        return lossName.ToLower() switch
        {
            "mse" or "mean_squared_error" => new MeanSquaredError(),
            "mae" or "mean_absolute_error" => new MeanAbsoluteError(),
            "rms" or "root_mean_squared_error" => new RootMeanSquaredError(),
            "cce" or "categorical_crossentropy" => new CategoricalCrossEntropy(),
            _ => throw new ArgumentException($"There is no loss function named {lossName}")
        };
    }

    public static Loss GetLoss(LossFunction lossName)
    {
        return lossName switch
        {
            LossFunction.MeanSquaredError => new MeanSquaredError(),
            LossFunction.MeanAbsoluteError => new MeanAbsoluteError(),
            LossFunction.RootMeanSquaredError => new RootMeanSquaredError(),
            LossFunction.CategoricalCrossEntropy => new CategoricalCrossEntropy()
        };
    }

    public virtual Tensor Derivate(Tensor ideal, Tensor predicted)
    {
        Tensor derivatives = new(predicted.shape);

        lossValue = 0;

        void derivationAction(int batch)
        {
            for (int i = 0; i < predicted.shape.nF1; i++)
            {
                derivatives[batch, i] = PartialDerivative(ideal[batch, i], predicted[batch, i]);
                lossValue += PartialError(ideal[batch, i], predicted[batch, i]);
            }
        }

        for (int i = 0; i < ideal.shape.n0; i++)
            derivationAction(i);

        lossValue = LossGen(lossValue, ideal.shape.nF1);
        lossValue /= ideal.shape.n0;

        return derivatives;
    }

    public virtual float ComputeError(Tensor ideal, Tensor predicted)
    {
        lossValue = 0;

        void errCalculatingAction(int batch)
        {
            for (int i = 0; i < ideal.shape.nF1; i++)
                lossValue += PartialError(ideal[batch, i], predicted[batch, i]);
        }

        for (int i = 0; i < ideal.shape.n0; i++)
            errCalculatingAction(i);

        lossValue = LossGen(lossValue, ideal.shape.n0);
        return lossValue;
    }

    protected virtual float PartialError(float ideal, float pred) => 0;
    protected virtual float PartialDerivative(float ideal, float pred) => 0;
    protected virtual float LossGen(float loss, float N) => 0;

    public static implicit operator Loss(string s) => Loss.GetLoss(s);
    public static implicit operator Loss(LossFunction l) => Loss.GetLoss(l);
}

[Serializable]
public unsafe class MeanSquaredError : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => MathF.Pow(ideal - pred, 2);
    protected sealed override float PartialDerivative(float ideal, float pred) => pred - ideal;
    protected sealed override float LossGen(float sum, float N) => sum / N;
}

[Serializable]
public unsafe class MeanAbsoluteError : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => MathF.Abs(ideal - pred);
    protected sealed override float PartialDerivative(float ideal, float pred) => -MathF.Sign(ideal - pred);
    protected sealed override float LossGen(float sum, float N) => sum / N;
}

[Serializable]
public unsafe class RootMeanSquaredError : Loss
{
    public sealed override Tensor Derivate(Tensor ideal, Tensor predicted)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.n0;

        float avCoeff = 1 / ideal.shape.nF1;

        Tensor derivatives = new(predicted.shape);

        float sum;

        for (int i = 0, j; i < actualMBSize; i++)
        {
            sum = 0;
            for (j = 0; j < predicted.shape.nF1; j++)
                sum += PartialError(ideal[i, j], predicted[i, j]);

            sum = MathF.Sqrt(avCoeff * sum);

            for (j = 0; j < predicted.shape.nF1; j++)
                derivatives[i, j] = -avCoeff * (ideal[i, j] - predicted[i, j]) / (sum + 1.0E-8F);

            lossValue += sum;
        }

        lossValue /= actualMBSize;

        return derivatives;
    }

    public sealed override float ComputeError(Tensor ideal, Tensor predicted)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.n0;

        float avCoeff = 1 / ideal.shape.nF1;

        float sum;

        for (int i = 0, j; i < actualMBSize; i++)
        {
            sum = 0;
            for (j = 0; j < predicted.shape.nF1; j++)
                sum += PartialError(ideal[i, j], predicted[i, j]);

            lossValue += MathF.Sqrt(avCoeff * sum);
        }

        lossValue /= actualMBSize;

        return lossValue;
    }

    protected sealed override float PartialError(float ideal, float pred) => MathF.Pow(ideal - pred, 2);
}


[Serializable]
public unsafe class CategoricalCrossEntropy : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => ideal * MathF.Log(pred + 1.0E-8F);
    protected sealed override float PartialDerivative(float ideal, float pred) => -ideal / (pred + 1.0E-8F);
    protected sealed override float LossGen(float sum, float N) => -sum / N;
}

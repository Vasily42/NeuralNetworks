namespace NeuralNetwork;

public unsafe abstract class Loss : Layer
{
    protected float lossValue;
    public float LossValue => lossValue;

    public Tensor Predicted => input;

    public override void Init(Optimizer optimizer)
    {
        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
    }

    public static Loss CreateLoss(string lossName) => lossName.ToLower() switch
    {
        "mae" => new MeanAbsoluteError(),
        "mse" => new MeanSquaredError(),
        "rms" => new RootMeanSquaredError(),
        "cross entropy" or "ce" => new CrossEntropy(),
        _ => throw new Exception(),
    };

    public sealed override void Forward(Tensor input, in int a, in bool t)
    {
        input.CopyTo(this.input);
    }

    public virtual void BackProp(Tensor ideal)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.batchSize;

        void derivationAction(int batch)
        {
            for (int i = 0; i < inputShape.flatBatchSize; i++)
            {
                this.inputDerivatives[batch, i] = PartialDerivative(ideal[batch, i], this.input[batch, i]);
                lossValue += PartialError(ideal[batch, i], this.input[batch, i]);
            }
        }

        for (int i = 0; i < actualMBSize; i++)
            derivationAction(i);

        lossValue = LossGen(lossValue);
        lossValue /= actualMBSize;

        prevLayer.BackProp(this.inputDerivatives, in actualMBSize);
    }

    public virtual float ComputeError(Tensor ideal)
    {
        lossValue = 0;

        void errCalculatingAction(int batch)
        {
            for (int i = 0; i < inputShape.flatBatchSize; i++)
                lossValue += PartialError(ideal[batch, i], this.input[batch, i]);
        }

        for (int i = 0; i < ideal.shape.batchSize; i++)
            errCalculatingAction(i);

        lossValue = LossGen(lossValue);
        return lossValue;
    }

    protected virtual float PartialError(float ideal, float pred) => 0;
    protected virtual float PartialDerivative(float ideal, float pred) => 0;
    protected virtual float LossGen(float loss) => 0;
}

public unsafe class MeanSquaredError : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => MathF.Pow(ideal - pred, 2);
    protected sealed override float PartialDerivative(float ideal, float pred) => pred - ideal;
    protected sealed override float LossGen(float sum) => sum / inputShape.flatBatchSize;
}

public unsafe class MeanAbsoluteError : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => MathF.Abs(ideal - pred);
    protected sealed override float PartialDerivative(float ideal, float pred) => -MathF.Sign(ideal - pred);
    protected sealed override float LossGen(float sum) => sum / inputShape.flatBatchSize;
}

public unsafe class RootMeanSquaredError : Loss
{
    public sealed override void BackProp(Tensor ideal)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.batchSize;

        float avCoeff = 1 / inputShape.flatBatchSize;

        float sum;

        for (int i = 0, j; i < actualMBSize; i++)
        {
            sum = 0;
            for (j = 0; j < inputShape.flatBatchSize; j++)
                sum += PartialError(ideal[i, j], input[i, j]);

            sum = MathF.Sqrt(avCoeff * sum);

            for (j = 0; j < inputShape.flatBatchSize; j++)
                inputDerivatives[i, j] = -avCoeff * (ideal[i, j] - input[i, j]) / (sum + epsilon);

            lossValue += sum;
        }

        lossValue /= actualMBSize;

        prevLayer.BackProp(this.inputDerivatives, in actualMBSize);
    }

    public sealed override float ComputeError(Tensor ideal)
    {
        lossValue = 0;

        int actualMBSize = ideal.shape.batchSize;

        float avCoeff = 1 / inputShape.flatBatchSize;

        float sum;

        for (int i = 0, j; i < actualMBSize; i++)
        {
            sum = 0;
            for (j = 0; j < inputShape.flatBatchSize; j++)
                sum += PartialError(ideal[i, j], input[i, j]);

            lossValue += MathF.Sqrt(avCoeff * sum);
        }

        lossValue /= actualMBSize;

        return lossValue;
    }

    protected sealed override float PartialError(float ideal, float pred) => MathF.Pow(ideal - pred, 2);
}



public unsafe class CrossEntropy : Loss
{
    protected sealed override float PartialError(float ideal, float pred) => ideal * MathF.Log(pred + epsilon);
    protected sealed override float PartialDerivative(float ideal, float pred) => -ideal / (pred + epsilon);
    protected sealed override float LossGen(float sum) => -sum;
}

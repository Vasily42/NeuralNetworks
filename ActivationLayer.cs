namespace NeuralNetwork;

public abstract unsafe class ActivationLayer : Layer
{
    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = inputShape;

        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);
    }

    public static ActivationLayer CreateActivation(string name) => name.ToLower() switch
    {
        "linear" => null,
        "sigmoid" => new Sigmoid(),
        "tangens" => new Tangens(),
        "relu" => new ReLU(),
        "elu" => new ELU(),
        "swish" => new Swish(),
        "softmax" => new Softmax(),
        _ => throw new Exception(),
    };

    protected override void ForwardAction(int batch)
    {
        for (int i = 0; i < inputShape.flatBatchSize; i++)
        {
            this.output[batch, i] = Activation(this.input[batch, i]);
        }
    }

    protected override void BackPropAction(int batch)
    {
        for (int i = 0; i < outputShape.flatBatchSize; i++)
            inputDerivatives[batch, i] = outputDerivatives[batch, i]
            * Derivative(this.output[batch, i], this.input[batch, i]);
    }

    protected virtual float Activation(float x) => 0;
    protected virtual float Derivative(float y, float x) => 0;
}

public class Sigmoid : ActivationLayer
{
    protected sealed override float Activation(float x) => 1f / (1 + MathF.Exp(-x));

    protected sealed override float Derivative(float y, float x) => ((1 - y) * y) + 0.02f;
}

public class Tangens : ActivationLayer
{
    protected sealed override float Activation(float x) => 2f / (1 + MathF.Exp(-2 * x)) - 1f;

    protected sealed override float Derivative(float y, float x)
    {
        return (1 - y * y) + 0.02f;
    }
}

public class ReLU : ActivationLayer
{
    protected sealed override float Activation(float x) => MathF.Max(0, x);

    protected sealed override float Derivative(float y, float x) => y > 0 ? 1 : 0;
}

public class ELU : ActivationLayer
{
    protected sealed override float Activation(float x) => x >= 0 ? x : 0.103f * (MathF.Exp(x) - 1);

    protected sealed override float Derivative(float y, float x) => y >= 0 ? 1 : y + 0.103f;
}

public class Swish : ActivationLayer
{
    protected sealed override float Activation(float x) => x * (1f / (1 + MathF.Exp(-x)));

    protected sealed override float Derivative(float y, float x) => (float)(y + 1f / (1 + MathF.Exp(-x)) * (1 - y));
}

public unsafe class Softmax : ActivationLayer
{
    protected sealed override void ForwardAction(int batch)
    {
        float sum = epsilon;
        for (int i = 0; i < inputShape.flatBatchSize; i++)
        {
            sum += MathF.Exp(input[batch, i]);
        }

        for (int i = 0; i < inputShape.flatBatchSize; i++)
        {
            output[batch, i] = MathF.Exp(input[batch, i] + epsilon) / sum;
        }
    }

    protected sealed override void BackPropAction(int batch)
    {
        float sum;

        for (int j = 0; j < inputShape.flatBatchSize; j++)
        {
            sum = 0;
            for (int i = 0; i < inputShape.flatBatchSize; i++)
            {
                if (i == j)
                {
                    sum += output[batch, i] * (1 - output[batch, j]) * outputDerivatives[batch, i];
                }
                else
                {
                    sum += -output[batch, i] * output[batch, j] * outputDerivatives[batch, i];
                }
            }
            outputDerivatives[batch, j] = sum;
        }
    }
}

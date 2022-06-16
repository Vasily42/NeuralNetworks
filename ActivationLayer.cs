namespace NeuralNetwork;

[Serializable]
public abstract unsafe class ActivationLayer : Layer
{
    public ActivationLayer(string name) : base(name) { }

    public override void Init(Optimizer optimizer)
    {
        outputShape = inputShape;

        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);
    }

    public static ActivationLayer GetActivationLayer(string functionName)
    {
        return functionName.ToLower() switch
        {
            "sigmoid" => new Sigmoid(),
            "tanh" => new Tanh(),
            "relu" => new ReLU(),
            "elu" => new ELU(),
            "swish" => new Swish(),
            "softmax" => new Softmax(),
            _ => throw new ArgumentException($"There is no function named {functionName} (activation function)")
        };
    }

    public static ActivationLayer GetActivationLayer(ActivationFunction functionName)
    {
        return functionName switch
        {
            ActivationFunction.Sigmoid => new Sigmoid(),
            ActivationFunction.Tanh => new Tanh(),
            ActivationFunction.ReLU => new ReLU(),
            ActivationFunction.ELU => new ELU(),
            ActivationFunction.Swish => new Swish(),
            ActivationFunction.Softmax => new Softmax()
        };
    }

    public override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        input.CopyTo(this.input);
        ForwardAction(actualMBSize);
        nextLayer.Forward(output, actualMBSize, training);
    }

    public override void BackProp(Tensor deriv, in int actualMBSize)
    {
        deriv.CopyTo(outputDerivatives);
        BackPropAction(actualMBSize);
        prevLayer.BackProp(inputDerivatives, in actualMBSize);
    }

    protected override void ForwardAction(int MBSize)
    {
        for (int i = 0; i < inputShape.nF1 * MBSize; i++)
        {
            this.output[i] = Activation(this.input[i]);
        }
    }

    protected override void BackPropAction(int MBSize)
    {
        for (int i = 0; i < outputShape.nF1 * MBSize; i++)
        {
            inputDerivatives[i] = outputDerivatives[i] * Derivative(this.output[i], this.input[i]);
        }
    }

    protected virtual float Activation(float x) => 0;
    protected virtual float Derivative(float y, float x) => 0;

    public static implicit operator ActivationLayer(string s) => ActivationLayer.GetActivationLayer(s);
    public static implicit operator ActivationLayer(ActivationFunction a) => ActivationLayer.GetActivationLayer(a);
}

[Serializable]
public class Sigmoid : ActivationLayer
{
    public Sigmoid(string name = null) : base(name)
    {
    }

    protected sealed override void ForwardAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = input.GetSpanVectors(out remain, 0, length);
        var bVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = Vector<float>.One / (Vector<float>.One + Tensor.Exp(-aVec[i]));
        }

        for (; remain < length; remain++) output[remain] = Activation(input[remain]);
    }

    protected sealed override void BackPropAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = outputDerivatives.GetSpanVectors(out remain, 0, length);
        var bVec = inputDerivatives.GetSpanVectors(0, length);
        var cVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = aVec[i] * ((Vector<float>.One - cVec[i]) * cVec[i]);
        }

        for (int i = remain; remain < length; i++) inputDerivatives[remain] = outputDerivatives[remain] * Derivative(output[remain], 0);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected sealed override float Activation(float x) => 1f / (1 + MathF.Exp(-x));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected sealed override float Derivative(float y, float x) => (1 - y) * y;
}

[Serializable]
public class Tanh : ActivationLayer
{
    public Tanh(string name = null) : base(name)
    {
    }

    protected sealed override void ForwardAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = input.GetSpanVectors(out remain, 0, length);
        var bVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = new Vector<float>(2f) / (Vector<float>.One + Tensor.Exp(aVec[i] * new Vector<float>(-2f))) - Vector<float>.One;
        }

        for (; remain < length; remain++) output[remain] = Activation(input[remain]);
    }

    protected sealed override void BackPropAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = outputDerivatives.GetSpanVectors(out remain, 0, length);
        var bVec = inputDerivatives.GetSpanVectors(0, length);
        var cVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = aVec[i] * (Vector<float>.One - cVec[i] * cVec[i]);
        }

        for (int i = remain; remain < length; i++) inputDerivatives[remain] = outputDerivatives[remain] * Derivative(output[remain], 0);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected sealed override float Activation(float x) => 2f / (1 + MathF.Exp(-2 * x)) - 1f;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected sealed override float Derivative(float y, float x) => 1 - y * y;
}

[Serializable]
public class ReLU : ActivationLayer
{
    public ReLU(string name = null) : base(name)
    {
    }

    protected sealed override void ForwardAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = input.GetSpanVectors(out remain, 0, length);
        var bVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = Vector.Max(aVec[i], Vector<float>.Zero);
        }

        for (; remain < length; remain++) output[remain] = Activation(input[remain]);
    }

    protected override void BackPropAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = outputDerivatives.GetSpanVectors(out remain, 0, length);
        var bVec = inputDerivatives.GetSpanVectors(0, length);
        var cVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = aVec[i] * Vector.ConditionalSelect(Vector.GreaterThan<float>(cVec[i], Vector<float>.Zero), Vector<float>.One, Vector<float>.Zero);
        }

        for (int i = remain; remain < length; i++) inputDerivatives[remain] = outputDerivatives[remain] * Derivative(output[remain], 0);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected sealed override float Activation(float x) => MathF.Max(x, 0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected sealed override float Derivative(float y, float x) => y > 0 ? 1 : 0;
}

[Serializable]
public class ELU : ActivationLayer
{
    private static readonly Vector<float> magicConst = new Vector<float>(0.103f);
    public ELU(string name = null) : base(name)
    {
    }

    protected sealed override void ForwardAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = input.GetSpanVectors(out remain, 0, length);
        var bVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = Vector.ConditionalSelect(Vector.GreaterThanOrEqual<float>(aVec[i], Vector<float>.Zero), aVec[i], magicConst * (Tensor.Exp(aVec[i]) - Vector<float>.One));
        }

        for (; remain < length; remain++) output[remain] = Activation(input[remain]);
    }

    protected override void BackPropAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = outputDerivatives.GetSpanVectors(out remain, 0, length);
        var bVec = inputDerivatives.GetSpanVectors(0, length);
        var cVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = aVec[i] * Vector.ConditionalSelect(Vector.GreaterThanOrEqual<float>(cVec[i], Vector<float>.Zero), Vector<float>.One, cVec[i] + magicConst);
        }

        for (int i = remain; remain < length; i++) inputDerivatives[remain] = outputDerivatives[remain] * Derivative(output[remain], 0);
    }

    protected sealed override float Activation(float x) => x >= 0 ? x : 0.103f * (MathF.Exp(x) - 1);

    protected sealed override float Derivative(float y, float x) => y >= 0 ? 1 : y + 0.103f;
}

[Serializable]
public class Swish : ActivationLayer
{
    public Swish(string name = null) : base(name)
    {
    }

    protected sealed override void ForwardAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = input.GetSpanVectors(out remain, 0, length);
        var bVec = output.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = aVec[i]  / (Vector<float>.One + Tensor.Exp(-aVec[i]));
        }

        for (; remain < length; remain++) output[remain] = Activation(input[remain]);
    }

    protected override void BackPropAction(int MBSize)
    {
        int remain;

        int length = MBSize * inputShape.nF1;

        var aVec = outputDerivatives.GetSpanVectors(out remain, 0, length);
        var bVec = inputDerivatives.GetSpanVectors(0, length);
        var cVec = output.GetSpanVectors(0, length);
        var dVec = input.GetSpanVectors(0, length);

        for (int i = 0; i < aVec.Length; i++)
        {
            bVec[i] = aVec[i] * (cVec[i]  + (Vector<float>.One - cVec[i]) / (Vector<float>.One + Tensor.Exp(-cVec[i])));
        }

        for (int i = remain; remain < length; i++) inputDerivatives[remain] = outputDerivatives[remain] * Derivative(output[remain], input[remain]);
    }

    protected sealed override float Activation(float x) => x / (1 + MathF.Exp(-x));

    protected sealed override float Derivative(float y, float x) => y + (1 - y) * (1 + MathF.Exp(-x));
}

[Serializable]
public unsafe class Softmax : ActivationLayer
{
    Tensor cache;

    public Softmax(string name = null) : base(name)
    {
    }

    public sealed override void Init(Optimizer optimizer)
    {
        base.Init(optimizer);

        cache = inputShape;
    }

    public sealed override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        input.CopyTo(this.input, actualMBSize);
        ParallelFor(ForwardAction, actualMBSize);
        nextLayer.Forward(this.output, in actualMBSize, in training);
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize)
    {
        deriv.CopyTo(outputDerivatives, actualMBSize);
        ParallelFor(BackPropAction, actualMBSize);
        prevLayer.BackProp(inputDerivatives, in actualMBSize);
    }

    protected sealed override void ForwardAction(int batch)
    {
        float sum = 0;
        float max = input.Max(batch);
        for (int i = 0; i < inputShape.nF1; i++)
        {
            cache[batch, i] = MathF.Exp(input[batch, i] - max);
            sum += cache[batch, i];
        }

        for (int i = 0; i < inputShape.nF1; i++)
        {
            output[batch, i] = cache[batch, i] / (sum + epsilon);
        }
    }

    protected sealed override void BackPropAction(int batch)
    {
        float sum;

        for (int j = 0; j < inputShape.nF1; j++)
        {
            sum = 0;
            for (int i = 0; i < inputShape.nF1; i++)
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
            inputDerivatives[batch, j] = sum;
        }
    }
}

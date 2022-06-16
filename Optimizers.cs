using System.Numerics;
using System.Xml.Schema;

namespace NeuralNetwork;

[Serializable]
public abstract class Optimizer
{
    internal float learningRate;

    public Optimizer(float learningRate)
    {
        this.learningRate = learningRate;
    }

    public static Optimizer GetOptimizer(string optimizerAlgorithm)
    {
        return optimizerAlgorithm.ToLower() switch
        {
            "sgd" or "none" => new SGD(),
            "momentum" => new SGD(momentum: 0.99f),
            "nesterov" => new SGD(momentum: 0.99f, nesterov: true),
            "adam" => new Adam(),
            "nadam" => new Adam(nesterov: true),
            _ => throw new ArgumentException($"There is no optimizer algorithm named {optimizerAlgorithm}")
        };
    }

    public static Optimizer GetOptimizer(OptimizerAlgorithm optimizerAlgorithm)
    {
        return optimizerAlgorithm switch
        {
            OptimizerAlgorithm.SGD => new SGD(),
            OptimizerAlgorithm.Momentum => new SGD(momentum: 0.99f),
            OptimizerAlgorithm.Nesterov => new SGD(momentum: 0.99f, nesterov: true),
            OptimizerAlgorithm.Adam => new Adam(),
            OptimizerAlgorithm.Nadam => new Adam(nesterov: true)
        };
    }

    public abstract void Init(int flatSize);

    public abstract void Reset();

    public abstract void Update(Tensor weights, Tensor gradient);

    public abstract Optimizer GetCopy();

    public static implicit operator Optimizer(string s) => Optimizer.GetOptimizer(s);
    public static implicit operator Optimizer(OptimizerAlgorithm o) => Optimizer.GetOptimizer(o);
}

[Serializable]
public unsafe class SGD : Optimizer
{
    private readonly float _momentum;
    private readonly bool _nesterov;
    private Tensor momentum;

    public SGD(float learningRate = 0.01f, float momentum = 0, bool nesterov = false) :
    base(learningRate)
    {
        this._momentum = momentum;
        if (momentum > 0)
            this._nesterov = nesterov;
        else this._nesterov = false;
    }

    public sealed override void Init(int flatSize) => this.momentum = new Tensor(flatSize).Fill(0);

    public sealed override void Reset() => momentum.Fill(0);
    public sealed override void Update(Tensor weights, Tensor gradient)
    {
        int remainA, remainB, remainC;

        var dVec = new Vector<float>(_momentum);
        var eVec = new Vector<float>(learningRate);

        var aVec = gradient.GetSpanVectors(out remainA);
        var bVec = momentum.GetSpanVectors(out remainB);
        var cVec = weights.GetSpanVectors(out remainC);

        if (_nesterov)
        {
            for (int i = 0; i < aVec.Length; i++)
            {
                bVec[i] = dVec * bVec[i] + eVec * aVec[i];
                cVec[i] -= dVec * bVec[i] + eVec * aVec[i];
                aVec[i] = Vector<float>.Zero;
            }

            for (int i = 0; i < weights.shape.nF0 % Tensor.vecCount; i++)
            {
                momentum[remainB + i] = _momentum * momentum[remainB + i] + learningRate * gradient[remainA + i];
                weights[remainC + i] -= _momentum * momentum[remainB + i] + learningRate * gradient[remainA + i];
                gradient[remainA + i] = 0;
            }
        }
        else
        {
            for (int i = 0; i < aVec.Length; i++)
            {
                bVec[i] = dVec * bVec[i] + eVec * aVec[i];
                cVec[i] -= bVec[i];
                aVec[i] = Vector<float>.Zero;
            }

            for (int i = 0; i < weights.shape.nF0 % Tensor.vecCount; i++)
            {
                momentum[remainB + i] = _momentum * momentum[remainB + i] + learningRate * gradient[remainA + i];
                weights[remainC + i] -= momentum[remainB + i];
                gradient[remainA + i] = 0;
            }
        }
    }

    public sealed override Optimizer GetCopy()
    {
        return new SGD(learningRate, _momentum, _nesterov);
    }
}

[Serializable]
public unsafe class Adam : Optimizer
{
    private const float epsilon = 1.0E-8F;

    private long iteration;
    private readonly float _momentum, _rmsCoeff;
    private readonly bool _nesterov;

    private Tensor firstMomentum, secondMomentum;

    public Adam(float learningRate = 0.001f, float momentum = 0.9f, float rmsCoeff = 0.999f, bool nesterov = false) :
    base(learningRate)
    {
        this._momentum = momentum;
        this._rmsCoeff = rmsCoeff;
        this._nesterov = nesterov;
        iteration = 1;
    }

    public sealed override void Init(int flatSize)
    {
        firstMomentum = new Tensor(flatSize).Fill(0);
        secondMomentum = new Tensor(flatSize).Fill(0);
    }

    public sealed override void Reset()
    {
        firstMomentum.Fill(0);
        secondMomentum.Fill(0);
    }

    public sealed override void Update(Tensor weights, Tensor gradient)
    {
        float firstUnbias, secondUnbias;

        if (_nesterov)
            for (int i = 0; i < weights.shape.nF0; i++)
            {
                firstMomentum[i] = _momentum * firstMomentum[i] + (1 - _momentum) * gradient[i];
                secondMomentum[i] = _rmsCoeff * secondMomentum[i] + (1 - _rmsCoeff) * gradient[i] * gradient[i];
                firstUnbias = firstMomentum[i] / (1 - MathF.Pow(_momentum, iteration));
                secondUnbias = secondMomentum[i] / (1 - MathF.Pow(_rmsCoeff, iteration));
                weights[i] -= learningRate / MathF.Sqrt(secondUnbias + epsilon) *
                (_momentum * firstUnbias + (1 - _momentum) * gradient[i] / (1 - MathF.Pow(_momentum, iteration)));
                gradient[i] = 0;
            }
        else for (int i = 0; i < weights.shape.nF0; i++)
            {
                firstMomentum[i] = _momentum * firstMomentum[i] + (1 - _momentum) * gradient[i];
                secondMomentum[i] = _rmsCoeff * secondMomentum[i] + (1 - _rmsCoeff) * gradient[i] * gradient[i];
                firstUnbias = firstMomentum[i] / (1 - MathF.Pow(_momentum, iteration));
                secondUnbias = secondMomentum[i] / (1 - MathF.Pow(_rmsCoeff, iteration));
                weights[i] -= learningRate * firstUnbias / MathF.Sqrt(secondUnbias + epsilon);
                gradient[i] = 0;
            }

        iteration++;
    }

    public sealed override Optimizer GetCopy()
    {
        return new Adam(learningRate, _momentum, _rmsCoeff, _nesterov);
    }
}

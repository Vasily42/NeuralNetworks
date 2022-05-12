namespace NeuralNetwork;

public abstract class Optimizer
{
    internal float learningRate;

    public Optimizer(float learningRate)
    {
        this.learningRate = learningRate;
    }

    public static Optimizer GetOptimizer(string optimizer) => optimizer.ToLower() switch
    {
        "none" => new SGD(),
        "momentum" => new SGD(momentum: 0.9f),
        "nesterov" => new SGD(momentum: 0.9f, nesterov: true),
        "adam" => new Adam(),
        "nadam" => new Adam(nesterov: true),
        _ => throw new Exception(),
    };

    public abstract void Init(int flatSize);

    public abstract void Reset();

    public abstract void Update(Tensor weights, Tensor gradient);

    public abstract Optimizer GetCopy();
}

public unsafe class SGD : Optimizer
{
    private readonly float _momentum;
    private readonly bool _nesterov;
    private Tensor momentum;

    public SGD(float learningRate = 0.001f, float momentum = 0, bool nesterov = false) :
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
        for (int i = 0; i < weights.shape.flatSize; i++)
        {
            momentum[i] = _momentum * momentum[i] + learningRate * gradient[i];
        }

        if (_nesterov)
            for (int i = 0; i < weights.shape.flatSize; i++)
            {
                weights[i] -= _momentum * momentum[i] + learningRate * gradient[i];
                gradient[i] = 0;
            }
        else
            for (int i = 0; i < weights.shape.flatSize; i++)
            {
                weights[i] -= momentum[i];
                gradient[i] = 0;
            }
    }

    public sealed override Optimizer GetCopy()
    {
        return new SGD(learningRate, _momentum, _nesterov);
    }
}

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
        for (int i = 0; i < weights.shape.flatSize; i++)
        {
            firstMomentum[i] = _momentum * firstMomentum[i] + (1 - _momentum) * gradient[i];
            secondMomentum[i] = _rmsCoeff * secondMomentum[i] + (1 - _rmsCoeff) * gradient[i] * gradient[i];
            firstUnbias = firstMomentum[i] / (1 - MathF.Pow(_momentum, iteration));
            secondUnbias = secondMomentum[i] / (1 - MathF.Pow(_rmsCoeff, iteration));
            weights[i] -= learningRate / MathF.Sqrt(secondUnbias + epsilon) * 
            (_momentum * firstUnbias + (1 - _momentum) * gradient[i] / (1 - MathF.Pow(_momentum, iteration)));
            gradient[i] = 0;
        }
        else for (int i = 0; i < weights.shape.flatSize; i++)
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

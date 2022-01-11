namespace NeuralNetwork;

public abstract class Optimizer
{
    public Model @base;

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

    public abstract void Update(ref Parameter p);
}

public unsafe class SGD : Optimizer
{
    private readonly float _momentum;
    private readonly bool _nesterov;

    public SGD(float learningRate = 0.001f, float momentum = 0, bool nesterov = false) :
    base(learningRate)
    {
        this._momentum = momentum;
        if (momentum > 0)
            this._nesterov = nesterov;
        else this._nesterov = false;
    }

    public override void Update(ref Parameter p)
    {
        p.firstMomentum = _momentum * p.firstMomentum + learningRate * p.gradient;
        if (_nesterov)
            p.value -= _momentum * p.firstMomentum + learningRate * p.gradient;
        else
            p.value -= p.firstMomentum;

        p.gradient = 0;
    }
}

public unsafe class Adam : Optimizer
{
    private const float epsilon = 1.0E-8F;
    private readonly float momentum, rmsCoeff;
    private readonly bool nesterov;

    public Adam(float learningRate = 0.001f, float momentum = 0.9f, float rmsCoeff = 0.999f, bool nesterov = false) :
    base(learningRate)
    {
        this.momentum = momentum;
        this.rmsCoeff = rmsCoeff;
        this.nesterov = nesterov;
    }

    public sealed override void Update(ref Parameter p)
    {
        float firstUnbias, secondUnbias;

        p.firstMomentum =
        momentum * p.firstMomentum +
        (1 - momentum) * p.gradient;

        p.secondMomentum =
        rmsCoeff * p.secondMomentum +
        (1 - rmsCoeff) * p.gradient * p.gradient;

        firstUnbias = (float)(p.firstMomentum
        / (1 - Math.Pow(momentum, @base.iteration + 1)));

        secondUnbias = (float)(p.secondMomentum
        / (1 - Math.Pow(rmsCoeff, @base.iteration + 1)));

        if (nesterov)
            p.value -= (float)(learningRate / Math.Sqrt(secondUnbias + epsilon)) * (float)
            (momentum * firstUnbias + ((1 - momentum) * p.gradient) / (1 - Math.Pow(momentum, @base.iteration + 1)));
        else
            p.value -= (float)(this.learningRate * firstUnbias / Math.Sqrt(secondUnbias + epsilon));

        p.gradient = 0;
    }
}

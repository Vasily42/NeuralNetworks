using System.Numerics;

namespace NeuralNetwork;

[Serializable]
public unsafe class Dense : Layer, IParameterized
{
    private Tensor kernel, kernelGradient, bias, biasGradient;

    Optimizer kernelOpt, biasOpt;

    Regularization kernelReg, biasReg;

    private readonly int numOfNeurons;

    protected Func<float> randomInitNum;

    protected readonly bool NonTrainable;

    public Dense(int numOfNeurons, ActivationLayer activationFunction = null, string parameterInitialization = "kaiming", Regularization kernelReg = null,
    Regularization biasReg = null, bool NonTrainable = false, string name = null) : base(name)
    {
        this.numOfNeurons = numOfNeurons;
        this.kernelReg = kernelReg;
        this.biasReg = biasReg;
        this.NonTrainable = NonTrainable;
        activationFunction?.Apply(this);
        randomInitNum = parameterInitialization switch
        {
            "xavier" => Xavier,
            "kaiming" => Kaiming
        };
    }

    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = inputShape.Change((1, numOfNeurons));

        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);

        bias = new Tensor(outputShape.n1).Clear();
        biasGradient = new Tensor(outputShape.n1).Clear();
        biasOpt = optimizer.GetCopy();
        biasOpt.Init(bias.shape.nF0);

        kernel = new Tensor(outputShape.n1, inputShape.n1).Fill(randomInitNum);
        kernelGradient = new Tensor(outputShape.n1, inputShape.n1).Clear();
        kernelOpt = optimizer.GetCopy();
        kernelOpt.Init(kernel.shape.nF0);
    }

    public void Reset()
    {
        bias.Clear();
        biasGradient.Clear();
        biasOpt.Reset();

        kernel.Fill(randomInitNum);
        kernelGradient.Clear();
        kernelOpt.Reset();
    }

    protected sealed override void ForwardAction(int batch)
    {
        float sum;

        for (int nthis = 0; nthis < outputShape.n1; nthis++)
        {
            sum = Tensor.Dot(input, kernel, inputShape.Point(batch, 0), kernel.shape.Point(nthis, 0), inputShape.nF1);
            sum += bias[nthis];
            this.output[batch, nthis] = sum;
        }
    }

    protected sealed override void BackPropAction(int batch)
    {
        int index = inputShape.Point(batch, 0);
        int length = inputShape.n1;

        inputDerivatives.Clear(index, length);

        int remainA, remainB;

        var aVec = inputDerivatives.GetSpanVectors(out remainA, index, length);
        var dVec = input.GetSpanVectors(index, length);

        for (int nthis = 0; nthis < outputShape.n1; nthis++)
        {
            index = kernel.shape.Point(nthis, 0);

            biasGradient[nthis] += outputDerivatives[batch, nthis];

            var bVec = kernel.GetSpanVectors(out remainB, index, length);
            var gVec = kernelGradient.GetSpanVectors(index, length);
            var cVec = new Vector<float>(outputDerivatives[batch, nthis]);

            for (int i = 0; i < bVec.Length; i++)
            {
                aVec[i] += bVec[i] * cVec;
                gVec[i] += dVec[i] * cVec;
            }

            for (int i = 0; i < inputShape.n1 % Tensor.vecCount; i++)
            {
                inputDerivatives[remainA + i] += outputDerivatives[batch, nthis] * kernel[remainB + i];
                kernelGradient[remainB + i] += input[remainA + i] * outputDerivatives[batch, nthis];
            }
        }
    }

    public void Correction()
    {
        if (NonTrainable) return;

        biasGradient /= inputShape.n0;

        biasReg?.GradPenalty(bias, biasGradient);
        biasOpt.Update(bias, biasGradient);

        kernelGradient /= inputShape.n0;

        kernelReg?.GradPenalty(kernel, kernelGradient);
        kernelOpt.Update(kernel, kernelGradient);

    }

    float Xavier() => MathF.Sqrt(6f / (inputShape.n1 + outputShape.n1)) * (2 * StGeneral.NextFloat() - 1);

    float Kaiming() => MathF.Sqrt(2f / inputShape.n1) * (2 * StGeneral.NextFloat() - 1);
}

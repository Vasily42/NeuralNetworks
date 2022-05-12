using System.Numerics;

namespace NeuralNetwork;

public unsafe class Dense : Layer, IParameterized
{
    private Tensor kernel, kernelGradient, bias, biasGradient;

    Optimizer kernelOpt, biasOpt;
    
    Regularization kernelReg, biasReg;

    private readonly int numOfNeurons;

    protected Func<float> randomInitNum;

    protected readonly bool NonTrainable;

    public Dense(int numOfNeurons, string activationFunction = "linear", string parameterInitialization = "kaiming", Regularization kernelReg = null,
    Regularization biasReg = null, bool NonTrainable = false)
    {
        this.numOfNeurons = numOfNeurons;
        this.kernelReg = kernelReg;
        this.biasReg = biasReg;
        this.NonTrainable = NonTrainable;
        InsertActivation(activationFunction);
        randomInitNum = parameterInitialization switch
        {
            "xavier" => Xavier,
            "kaiming" => Kaiming
        };
    }

    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = inputShape.NeuralChange(xLength: numOfNeurons);

        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);

        bias = new Tensor(outputShape.xLength).Fill(0);
        biasGradient = new Tensor(outputShape.xLength).Fill(0);
        biasOpt = optimizer.GetCopy();
        biasOpt.Init(bias.shape.flatSize);

        kernel = new Tensor(inputShape.xLength, outputShape.xLength).Fill(randomInitNum);
        kernelGradient = new Tensor(inputShape.xLength, outputShape.xLength).Fill(0);
        kernelOpt = optimizer.GetCopy();
        kernelOpt.Init(kernel.shape.flatSize);
    }

    public void Reset()
    {
        bias.Fill(0);
        biasGradient.Fill(0);
        biasOpt.Reset();

        kernel.Fill(randomInitNum);
        kernelGradient.Fill(0);
        kernelOpt.Reset();
    }

    protected sealed override void ForwardAction(int batch)
    {
        float sum;

        var vecLen = Vector<float>.Count;

        int iB, i = 0;

        for (int nthis = 0; nthis < outputShape.xLength; nthis++)
        {
            sum = 0;

            for (int nnext = 0; nnext < inputShape.xLength; nnext++)
            {
                sum += this.input[batch, nnext] * kernel[nnext, nthis];
            }

            sum += bias[nthis];
            this.output[batch, nthis] = sum;
        }
    }

    protected sealed override void BackPropAction(int batch)
    {
        for (int nthis = 0; nthis < outputShape.xLength; nthis++)
        {
            biasGradient[nthis] += outputDerivatives[batch, nthis];
        }

        for (int nnext = 0; nnext < inputShape.xLength; nnext++)
        {
            inputDerivatives[batch, nnext] = 0;
            for (int nthis = 0; nthis < outputShape.xLength; nthis++)
            {
                kernelGradient[nnext, nthis] += this.input[batch, nnext] * outputDerivatives[batch, nthis];
                inputDerivatives[batch, nnext] += kernel[nnext, nthis] * outputDerivatives[batch, nthis];
            }
        }
    }

    public void Correction()
    {
        if (NonTrainable) return;

        biasGradient /= inputShape.batchSize;

        biasReg?.GradPenalty(bias, biasGradient);
        biasOpt.Update(bias, biasGradient);

        kernelGradient /= inputShape.batchSize;

        kernelReg?.GradPenalty(kernel, kernelGradient);
        kernelOpt.Update(kernel, kernelGradient);

    }

    float Xavier() => MathF.Sqrt(6f / (inputShape.xLength + outputShape.xLength)) * (2 * StGeneral.NextFloat() - 1);

    float Kaiming() => MathF.Sqrt(2f / inputShape.xLength) * (2 * StGeneral.NextFloat() - 1);
}

using System.ComponentModel.DataAnnotations;
using System.Security.AccessControl;

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

        for (int NThis = 0; NThis < outputShape.xLength; NThis++)
        {
            sum = 0;

            for (int NNext = 0; NNext < inputShape.xLength; NNext++)
            {
                sum += this.input[batch, NNext] * kernel[NNext, NThis];
            }

            sum += bias[NThis];
            this.output[batch, NThis] = sum;
        }
    }

    protected sealed override void BackPropAction(int batch)
    {
        for (int NThis = 0; NThis < outputShape.xLength; NThis++)
        {
            biasGradient[NThis] += outputDerivatives[batch, NThis];
        }

        for (int NNext = 0; NNext < inputShape.xLength; NNext++)
        {
            inputDerivatives[batch, NNext] = 0;
            for (int NThis = 0; NThis < outputShape.xLength; NThis++)
            {
                kernelGradient[NNext, NThis] += this.input[batch, NNext] * outputDerivatives[batch, NThis];
                inputDerivatives[batch, NNext] += kernel[NNext, NThis] * outputDerivatives[batch, NThis];
            }
        }
    }

    public void Correction()
    {
        if (NonTrainable) return;

        biasReg?.GradPenalty(bias, biasGradient);
        biasOpt.Update(bias, biasGradient);

        kernelReg?.GradPenalty(kernel, kernelGradient);
        kernelOpt.Update(kernel, kernelGradient);

    }

    float Xavier() => MathF.Sqrt(6f / (inputShape.xLength + outputShape.xLength)) * (2 * StGeneral.NextFloat() - 1);

    float Kaiming() => MathF.Sqrt(2f / inputShape.xLength) * (2 * StGeneral.NextFloat() - 1);
}

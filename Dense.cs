using System.ComponentModel.DataAnnotations;

namespace NeuralNetwork;

public unsafe class Dense : CalcLayer
{
    private Tensor kernel, kernelGradient, bias, biasGradient;

    Optimizer kernelOpt, biasOpt;

    Regularization? kernelReg, biasReg;

    private readonly int numOfNeurons;

    public Dense(int numOfNeurons, string activationFunction = "linear", string parameterInitialization = "kaiming", Regularization? kernelReg = null,
    Regularization? biasReg = null, bool NonTrainable = false) :
    base(activationFunction, parameterInitialization, NonTrainable)
    {
        this.numOfNeurons = numOfNeurons;
        this.kernelReg = kernelReg;
        this.biasReg = biasReg;
    }

    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = inputShape.NeuralChange(xLength: numOfNeurons);

        base.fanIn = inputShape.xLength;
        base.fanOut = outputShape.xLength;

        input = Tensor.Create(inputShape);
        inputDerivatives = Tensor.Create(inputShape);
        output = Tensor.Create(outputShape);
        outputDerivatives = Tensor.Create(outputShape);

        bias = new Tensor1(outputShape.xLength).Fill(0);
        biasGradient = new Tensor1(outputShape.xLength).Fill(0);
        biasOpt = optimizer.GetCopy();
        biasOpt.Init(bias.shape.flatSize);

        kernel = new Tensor2(inputShape.xLength, outputShape.xLength).Fill(randomInitNum);
        kernelGradient = new Tensor2(inputShape.xLength, outputShape.xLength).Fill(0);
        kernelOpt = optimizer.GetCopy();
        kernelOpt.Init(kernel.shape.flatSize);
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

    public sealed override void Correction()
    {
        if (NonTrainable) return;
        
        biasReg?.GradPenalty(bias, biasGradient);
        biasOpt.Update(bias, biasGradient);

        kernelReg?.GradPenalty(kernel, kernelGradient);
        kernelOpt.Update(kernel, kernelGradient);
        
    }
}

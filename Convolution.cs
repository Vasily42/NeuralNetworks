namespace NeuralNetwork;

public unsafe class Convolution2D : Layer, IParameterized
{
    private Tensor kernel, kernelGradient, bias, biasGradient;

    Optimizer kernelOpt, biasOpt;

    Regularization kernelReg, biasReg;

    private readonly int numberOfFilters;
    internal readonly (byte x, byte y) strides, kernelSize;
    private readonly Padding2D paddingLayer;

    protected Func<float> randomInitNum;

    protected readonly bool NonTrainable;

    public Convolution2D(int numberOfFilters,
    (byte x, byte y) kernelSize,
    (byte x, byte y) strides,
    string activationFunction = "linear",
    string padding = "same",
    string parameterInitialization = "kaiming",
    Regularization kernelReg = null,
    Regularization biasReg = null,
    bool NonTrainable = false)
    {
        this.numberOfFilters = numberOfFilters;
        this.kernelSize = kernelSize;
        this.strides = strides;
        if (padding != "valid")
        {
            paddingLayer = new Padding2D(padding, kernelSize, strides);
            Apply(paddingLayer);
        }
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
        int outXLength = (int)((inputShape.xLength - kernelSize.x) / (float)strides.x + 1);
        int outYLength = (int)((inputShape.yLength - kernelSize.y) / (float)strides.y + 1);

        outputShape = inputShape.NeuralChange(
        channels: numberOfFilters, xLength: outXLength, yLength: outYLength);

        inputDerivatives = new Tensor(inputShape);
        input = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);

        bias = new Tensor(outputShape.channels).Fill(0);
        biasGradient = new Tensor(outputShape.channels).Fill(0);
        biasOpt = optimizer.GetCopy();
        biasOpt.Init(bias.shape.flatSize);

        kernel = new Tensor(numberOfFilters, inputShape.channels, kernelSize.y, kernelSize.x).Fill(randomInitNum);
        kernelGradient = new Tensor(numberOfFilters, inputShape.channels, kernelSize.y, kernelSize.x).Fill(0);
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
        float sum = 0;
        for (int filter = 0; filter < outputShape.channels; filter++)
            for (int iOut = 0; iOut < outputShape.yLength; iOut++)
                for (int jOut = 0; jOut < outputShape.xLength; jOut++)
                {
                    this.output[batch, filter, iOut, jOut] = bias[filter];
                }

        for (int filter = 0; filter < numberOfFilters; filter++)
            for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
            {
                for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
                    for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
                    {
                        for (int y = 0; y < kernelSize.y; y++)
                            for (int x = 0; x < kernelSize.x; x++)
                            {
                                sum += kernel[filter, inputChannel, y, x] *
                                this.input[batch, inputChannel, i + y, j + x];
                            }
                        this.output[batch, filter, iOut, jOut] += sum;
                        sum = 0;
                    }
            }
    }

    protected sealed override void BackPropAction(int batch)
    {
        for (int filter = 0; filter < numberOfFilters; filter++)
            for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
            {
                for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
                    for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
                    {
                        for (int y = 0; y < kernelSize.y; y++)
                            for (int x = 0; x < kernelSize.x; x++)
                            {
                                kernelGradient[filter, inputChannel, y, x] += outputDerivatives[batch, filter, iOut, jOut] * this.input[batch, inputChannel, i + y, j + x];
                                inputDerivatives[batch, inputChannel, i + y, j + x] += outputDerivatives[batch, filter, iOut, jOut] * kernel[filter, inputChannel, y, x];
                            }
                    }
            }

        for (int filter = 0; filter < outputShape.channels; filter++)
            for (int iOut = 0; iOut < outputShape.yLength; iOut++)
                for (int jOut = 0; jOut < outputShape.xLength; jOut++)
                {
                    biasGradient[filter] += outputDerivatives[batch, filter, iOut, jOut];
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

    float Xavier() => MathF.Sqrt(6f / (kernelSize.x * kernelSize.y * inputShape.channels + kernelSize.x * kernelSize.y * outputShape.channels) * (2 * StGeneral.NextFloat() - 1));
    float Kaiming() => MathF.Sqrt(2f / (kernelSize.x * kernelSize.y * inputShape.channels)) * (2 * StGeneral.NextFloat() - 1);
}

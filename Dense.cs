namespace NeuralNetwork;

public unsafe class Dense : CalcLayer
{
    private Parameter[] bias;
    private Parameter[][] weights;

    private readonly int numOfNeurons;

    public Dense(int numOfNeurons, string activationFunction = "linear", string parameterInitialization = "kaiming", bool NonTrainable = false) :
    base(activationFunction, parameterInitialization, NonTrainable)
    {
        this.numOfNeurons = numOfNeurons;
    }

    public sealed override void Init()
    {
        outputShape = inputShape.Change(xLength: numOfNeurons);

        base.fanIn = inputShape.xLength;
        base.fanOut = outputShape.xLength;

        input = Tensor.Create(inputShape);

        inputDerivatives = Tensor.Create(inputShape);

        output = Tensor.Create(outputShape);

        outputDerivatives = Tensor.Create(outputShape);

        bias = new Parameter[outputShape.xLength];

        weights = new Parameter[inputShape.xLength][];

        for (int NThis = 0; NThis < inputShape.xLength; NThis++)
        {
            weights[NThis] = new Parameter[outputShape.xLength];
            for (int NNext = 0; NNext < outputShape.xLength; NNext++)
                weights[NThis][NNext] = new Parameter(randomInitNum());
        }

        for (int NNext = 0; NNext < outputShape.xLength; NNext++)
            bias[NNext] = new Parameter(0);
    }

    protected sealed override void ForwardAction(int batch)
    {
        float sum;

        for (int NThis = 0; NThis < outputShape.xLength; NThis++)
        {
            sum = 0;

            for (int NNext = 0; NNext < inputShape.xLength; NNext++)
            {
                sum += this.input[batch, NNext] * weights[NNext][NThis].value;
            }

            sum += bias[NThis].value;
            this.output[batch, NThis] = sum;
        }
    }

    protected sealed override void BackPropAction(int batch)
    {
        for (int NThis = 0; NThis < outputShape.xLength; NThis++)
        {
            bias[NThis].gradient += outputDerivatives[batch, NThis];
        }

        for (int NNext = 0; NNext < inputShape.xLength; NNext++)
        {
            inputDerivatives[batch, NNext] = 0;
            for (int NThis = 0; NThis < outputShape.xLength; NThis++)
            {
                weights[NNext][NThis].gradient += this.input[batch, NNext] * outputDerivatives[batch, NThis];
                inputDerivatives[batch, NNext] += weights[NNext][NThis].value * outputDerivatives[batch, NThis];
            }
        }
    }

    public sealed override void Correction(Optimizer optimizer, Regularization regularizer)
    {
        if (NonTrainable) return;
        for (int NThis = 0; NThis < outputShape.xLength; NThis++)
        {
            regularizer?.GradPenalty(ref bias[NThis]);
            optimizer.Update(ref bias[NThis]);
        }

        for (int NNext = 0; NNext < inputShape.xLength; NNext++)
        {
            for (int NThis = 0; NThis < outputShape.xLength; NThis++)
            {
                regularizer?.GradPenalty(ref weights[NNext][NThis]);
                optimizer.Update(ref weights[NNext][NThis]);
            }
        }
    }
}

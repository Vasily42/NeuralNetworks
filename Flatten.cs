namespace NeuralNetwork;

public unsafe class Flatten : Layer
{
    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = new Tensor.ShapeInfo(inputShape.batchSize, inputShape.flatBatchSize);
        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);
    }

    protected sealed override void ForwardAction(int batch)
    {
        for (int i = 0; i < outputShape.flatBatchSize; i++)
            output[batch, i] = input[batch, i];
    }

    protected sealed override void BackPropAction(int batch)
    {
        for (int i = 0; i < inputShape.flatBatchSize; i++)
            inputDerivatives[batch, i] = outputDerivatives[batch, i];
    }
}

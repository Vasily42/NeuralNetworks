namespace NeuralNetwork;

public unsafe class Input : Layer
{
    public Input(Tensor.ShapeInfo shape)
    {
        this.inputShape = shape;
        outputShape = inputShape;
    }

    public void Forward(Tensor input, bool training)
    {
        int actualMBSize = input.shape.batchSize;
        if (input.shape.rank == inputShape.rank)
            nextLayer.Forward(input, in actualMBSize, in training);
        else nextLayer.Forward(Tensor.Reshape(inputShape, input), in actualMBSize, in training);
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize) { }
}

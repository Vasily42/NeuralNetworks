namespace NeuralNetwork;

public unsafe class Input : Layer
{
    public event Action<Tensor> OnBackPropEnd;
    public Input(Tensor.ShapeInfo shape)
    {
        this.inputShape = shape;
        outputShape = inputShape;
    }

    public override void Init(Optimizer optimizer)
    {
        inputDerivatives = inputShape;
    }

    public void Forward(Tensor input, bool training)
    {
        int actualMBSize = input.shape.batchSize;
        if (input.shape.flatSize % inputShape.flatBatchSize == 0)
            nextLayer.Forward(input, in actualMBSize, in training);
        else throw new Exception("wrong format of input");
    }

    public void Forward(Array input, bool training)
    {
        Forward(Tensor.Create(input).Reshape(inputShape), training);
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize) 
    {
        deriv.CopyTo(inputDerivatives);

        OnBackPropEnd?.Invoke(inputDerivatives);
    }
}

public unsafe class Output : Layer
{
    public sealed override Tensor OutputTensor => output;

    public event Action<Tensor> OnForwardEnd;
    
    public sealed override void Init(Optimizer optimizer)
    {
        output = new(inputShape);
    }

    public sealed override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        input.CopyTo(output);
        
        OnForwardEnd?.Invoke(output);
    }

    public void BackProp(Tensor deriv)
    {
        int actualMBSize = deriv.shape.batchSize;
        if (deriv.shape.rank == inputShape.rank)
            prevLayer.BackProp(deriv, in actualMBSize);
        else prevLayer.BackProp(deriv.Reshape(inputShape), actualMBSize);
    }
}

namespace NeuralNetwork;

[Serializable]
public unsafe class Input : Layer
{
    public event Action<Tensor> OnBackPropEnd;
    public Input(Tensor.ShapeInfo shape, string name = null) : base(name)
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
        int actualMBSize = input.shape.n0;
        if (input.shape.nF0 % inputShape.nF1 == 0)
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

[Serializable]
public unsafe class Output : Layer
{
    public Output(string name = null) : base(name)
    {
    }

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
        int actualMBSize = deriv.shape.n0;
        if (deriv.shape.rank == inputShape.rank)
            prevLayer.BackProp(deriv, in actualMBSize);
        else prevLayer.BackProp(deriv.Reshape(inputShape), actualMBSize);
    }
}

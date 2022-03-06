namespace NeuralNetwork;

public abstract class Layer
{
    protected const float epsilon = 1.0E-8F;

    internal Layer nextLayer, prevLayer;

    protected Tensor input, output, inputDerivatives, outputDerivatives;

    internal Tensor.ShapeInfo inputShape, outputShape;

    internal Layer Leave => nextLayer == null ? this : nextLayer.Leave;

    internal Layer Root => prevLayer == null ? this : prevLayer.Root;

    public Layer Apply(Layer prevLayer)
    {
        var temp = prevLayer.Leave;

        temp.nextLayer = Root;

        Root.prevLayer = temp;

        return Leave;
    }

    public virtual void Init(Optimizer optimizer) { }

    public void InitGraph(Optimizer optimizer)
    {
        if (prevLayer != null)
        {
            this.inputShape = prevLayer.outputShape;
        }

        Init(optimizer);
        //Console.WriteLine(nextLayer);
        nextLayer?.InitGraph(optimizer);
    }

    public void ParameterCorrection()
    {
        if (this is CalcLayer calc)
        {
            calc.Correction();
        }
        nextLayer?.ParameterCorrection();
    }

    protected void InsertAhead(Layer layer) => layer.Apply(this);

    protected void InsertActivation(string activationName = null)
    {
        if (activationName != null && activationName.ToLower() != "linear")
            InsertAhead(ActivationLayer.CreateActivation(activationName));
    }

    //	protected virtual void WriteWeights(BinaryWriter writer);

    //	protected virtual void WriteModel(BinaryWriter writer);

    //	protected abstract void ReadWeights(BinaryReader reader);

    //	public abstract void ReadModel(BinaryReader reader);

    public virtual void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        input.CopyTo(this.input, actualMBSize);
        for (int i = 0; i < actualMBSize; i++)
            ForwardAction(i);
        nextLayer?.Forward(this.output, in actualMBSize, in training);
    }

    protected virtual void ForwardAction(int batch) { }

    public virtual void BackProp(Tensor deriv, in int actualMBSize)
    {
        deriv.CopyTo(outputDerivatives, actualMBSize);
        for (int i = 0; i < actualMBSize; i++)
            BackPropAction(i);
        prevLayer?.BackProp(inputDerivatives, in actualMBSize);
    }

    protected virtual void BackPropAction(int batch) { }
}

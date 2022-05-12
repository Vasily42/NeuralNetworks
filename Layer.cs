namespace NeuralNetwork;

public abstract class Layer
{
    protected const float epsilon = 1.0E-8F;

    protected bool initialized = false;

    internal Layer nextLayer, prevLayer;

    internal Tensor input, output, inputDerivatives, outputDerivatives;

    public virtual Tensor OutputTensor => output.GetCopy();

    internal Tensor.ShapeInfo inputShape, outputShape;

    public Layer Tail => nextLayer == null ? this : nextLayer.Tail;

    public Layer Head => prevLayer == null ? this  : prevLayer.Head;

    internal void ConnectTo(dynamic nextBlock)
    {
        if (nextBlock is Array) ConnectToArray(nextBlock);
        else ConnectToSeq(nextBlock);
    }

    protected virtual void ConnectToSeq(Layer nextLayer)
    {
        this.nextLayer = nextLayer;
    }

    protected virtual void ConnectToArray(Layer[] nextLayers)
    {
        Copy copyLayer = new();

        copyLayer.ConnectToArray(nextLayers);

        ConnectTo(copyLayer);
    }

    public Layer Apply(dynamic prevBlock)
    {
        if (prevBlock is Array) return Head.ApplyArray(prevBlock).Tail;
        else return Head.ApplySeq(prevBlock.Tail).Tail;
    }

    internal virtual Layer ApplySeq(Layer prevLayer)
    {
        this.prevLayer = prevLayer;

        prevLayer.ConnectTo(this);

        return this;
    }

    internal virtual Layer ApplyArray(Layer[] prevLayers)
    {
        Layer concat = new Concat();

        concat.ApplyArray(prevLayers);

        ApplySeq(concat);

        return this;
    }

    public virtual void Init(Optimizer optimizer) { }

    public virtual void InitGraph(Optimizer optimizer = null, LayerCommander commander = null)
    {
        if (!initialized)
        {
            initialized = true;
            commander?.AddLayer(this);
            if (prevLayer != null)
            {
                this.inputShape = prevLayer.outputShape;
            }

            Init(optimizer);
        }
        nextLayer?.InitGraph(optimizer, commander);
    }

    public virtual void Reset()
    {
        if (this is IParameterized parameterized) parameterized.Reset();
    }

    public virtual void ParameterCorrection()
    {
        if (this is IParameterized parameterized) parameterized.Correction();
    }

    protected void InsertActivation(string activationName = null)
    {
        if (activationName != null && activationName.ToLower() != "linear")
            ActivationLayer.CreateActivation(activationName).Apply(this);
    }

    //	protected virtual void WriteWeights(BinaryWriter writer);

    //	protected virtual void WriteModel(BinaryWriter writer);

    //	protected abstract void ReadWeights(BinaryReader reader);

    //	public abstract void ReadModel(BinaryReader reader);

    public virtual void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        input.CopyTo(this.input, actualMBSize);
        Parallel.For(0, actualMBSize, ForwardAction);
        nextLayer?.Forward(this.output, in actualMBSize, in training);
    }

    protected virtual void ForwardAction(int batch) { }

    public virtual void BackProp(Tensor deriv, in int actualMBSize)
    {
        deriv.CopyTo(outputDerivatives, actualMBSize);
        Parallel.For(0, actualMBSize, BackPropAction);
        prevLayer?.BackProp(inputDerivatives, in actualMBSize);
    }

    protected virtual void BackPropAction(int batch) { }

    public static Layer[] operator +(Layer a, Layer b)
    {
        return new Layer[2] { a, b };
    }

    public static Layer[] operator +(Layer[] arr, Layer a)
    {
        Layer[] newArr = new Layer[arr.Length + 1];
        for (int i = 0; i < arr.Length; i++) newArr[i] = arr[i];
        newArr[^1] = a;
        return newArr;
    }

    public static Layer operator /(Layer a, Layer b)
    {
        return b.ApplySeq(a);
    }

    public static Layer operator /(Layer[] arr, Layer a)
    {
        return a.ApplyArray(arr);
    }

    public static Layer[] operator /(Layer a, Layer[] arr)
    {
        Copy c = new();

        for (int i = 0; i < arr.Length; i++) arr[i].ApplySeq(c);

        c.ApplySeq(a);

        return arr;
    }
}

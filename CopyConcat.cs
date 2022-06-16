namespace NeuralNetwork;

[Serializable]
public class Copy : Layer
{
    internal Layer[] nextLayers;
    internal new Tensor[] outputDerivatives;

    private int pulled;

    public Copy(string name = null) : base(name)
    {
    }

    protected sealed override void ConnectToSeq(Layer nextLayer)
    {
        if (nextLayers == null)
        {
            nextLayers = new Layer[1] { nextLayer };
            return;
        }

        Layer[] newNextLayers = new Layer[nextLayers.Length + 1];

        for (int i = 0; i < nextLayers.Length; i++) newNextLayers[i] = nextLayers[i];

        newNextLayers[^1] = nextLayer;

        nextLayers = newNextLayers;
    }

    public sealed override void Init(Optimizer optimizer)
    {
        outputShape = inputShape;
        inputDerivatives = new Tensor(inputShape);
        outputDerivatives = new Tensor[nextLayers.Length];
        for (int i = 0; i < nextLayers.Length; i++)
        {
            outputDerivatives[i] = new Tensor(inputShape);
        }
    }

    public sealed override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        for (int i = 0; i < nextLayers.Length; i++) nextLayers[i].Forward(input, in actualMBSize, in training);
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize)
    {
        lock (outputDerivatives)
        {
            outputDerivatives[pulled] = deriv;
            Interlocked.Increment(ref pulled);
        }
        if (pulled == nextLayers.Length)
        {
            inputDerivatives.Fill(0);
            for (int i = 0; i < outputDerivatives.Length; i++)
                for (int j = 0; j < inputDerivatives.shape.n0; j++) inputDerivatives[j] += outputDerivatives[i][j];

            pulled = 0;

            prevLayer.BackProp(inputDerivatives, in actualMBSize);
        }
    }

    public sealed override void InitGraph(Optimizer optimizer, LayerCommander commander)
    {
        if (prevLayer != null)
        {
            this.inputShape = prevLayer.outputShape;
        }

        Init(optimizer);

        for(int i = 0; i < nextLayers.Length; i++) nextLayers[i].InitGraph(optimizer, commander);
    }

    
}

public class Concat : Layer
{
    internal Tensor.ShapeInfo[] inputShapes;
    private int[] modShape;
    private int outConcatFlatSize;
    private int nFRev;
    internal Layer[] prevLayers;
    internal new Tensor[] inputDerivatives, inputs;

    private int pulled, axis;

    public Concat(string name = null) : base(name)
    {
    }

    internal sealed override Layer ApplySeq(Layer prevLayer)
    {
        if (prevLayers == null)
        {
            prevLayers = new Layer[1] { prevLayer };
            return this;
        }

        Layer[] newPrevLayers = new Layer[prevLayers.Length + 1];

        for (int i = 0; i < prevLayers.Length; i++) newPrevLayers[i] = prevLayers[i];

        newPrevLayers[^1] = prevLayer;

        prevLayers = newPrevLayers;

        prevLayer.ConnectTo(this);

        return this;
    }
    
    internal sealed override Layer ApplyArray(Layer[] prevLayers)
    {
        if (this.prevLayers == null)
        {
            this.prevLayers = prevLayers;
            for (int j = 0; j < prevLayers.Length; j++) prevLayers[j].ConnectTo(this);
            return this;
        }

        Layer[] newPrevLayers = new Layer[prevLayers.Length + this.prevLayers.Length];

        int i = 0;
        for (; i < this.prevLayers.Length; i++) newPrevLayers[i] = this.prevLayers[i];
        for (int j = 0; j < prevLayers.Length; j++, i++) 
        {
            newPrevLayers[i] = prevLayers[j];
            prevLayers[j].ConnectTo(this);
        }

        this.prevLayers = newPrevLayers;

        return this;
    }

    public sealed override void Init(Optimizer optimizer)
    {
        inputShapes = prevLayers.Select(x => x.outputShape).ToArray();
        inputs = new Tensor[inputShapes.Length];
        inputDerivatives = new Tensor[inputShapes.Length];
        for (int i = 0; i < inputShapes[0].rank; i++)
        {
            if (inputShapes[0][i] != inputShapes[1][i])
            {
                axis = i;
                break;
            }
        }

        modShape = new int[inputShapes.Length];

        outConcatFlatSize = 0;

        for (int i = 0; i < prevLayers.Length; i++)
        {
            inputs[i] = new(inputShapes[i]);
            inputDerivatives[i] = new(inputShapes[i]);
            modShape[i] = inputShapes[i].NF[axis];
            outConcatFlatSize += modShape[i];
        }

        outputShape = inputShapes[0].Change((axis, inputShapes.Select(x => x[axis]).Sum()));

        nFRev = inputs[0].shape.n0 / modShape[0];

        output = new(outputShape);
    }

    public sealed override void Forward(Tensor input, in int actualMBSize, in bool training)
    {
        lock (this)
        {
            inputs[pulled] = input;
            Interlocked.Increment(ref pulled);
        }
        if (pulled == prevLayers.Length)
        {
            pulled = 0;

            for (int i = 0; i < nFRev; i++)
                for (int j = 0, k = 0; j < inputs.Length; j++)
                    for (int q = 0; q < modShape[j]; q++, k++) output[i * outConcatFlatSize + k] = inputs[j][i * modShape[j] + q];

            nextLayer?.Forward(output, in actualMBSize, in training);
        }
    }

    public sealed override void BackProp(Tensor deriv, in int actualMBSize)
    {
        for (int i = 0; i < nFRev; i++)
            for (int j = 0, k = 0; j < inputs.Length; j++)
                for (int q = 0; q < modShape[j]; q++, k++) inputDerivatives[j][i * modShape[j] + q] = outputDerivatives[i * outConcatFlatSize + k];

        for (int i = 0; i < inputs.Length; i++) prevLayers[i].BackProp(inputDerivatives[i], in actualMBSize);
    }
}


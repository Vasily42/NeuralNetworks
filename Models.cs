namespace NeuralNetwork;

public abstract class Model
{
    public bool Initialized { get; protected set; }

    protected LayerCommander commander;

    internal long iteration;
    public long Iteration => iteration;

    public bool Training { get; set; }

    internal readonly Random helpRnd;

    public Model()
    {
        helpRnd = new Random();
    }
}

// public sealed class NonSequential : Model
// {
//     public event Action<float> ErrorRendering, IterationRendering, EpochRendering;

//     private Input[] inputs;
//     private Output[] outputs;
//     private Loss[] losses;

//     public void Init(Input[] inputs, Output[] outputs, Optimizer optimizer, Loss[] losses)
//     {
//         this.inputs = inputs;
//         this.outputs = outputs;

//         this.losses = losses;

//         commander = new LayerCommander();

//         for (int i = 0; i < inputs.Length; i++) inputs[i].InitGraph(optimizer, commander);
//     }

//     public void ForwardBatch(Tensor[] forwardData)
//     {

//     }
// }

public sealed class Sequential : Model
{
    public Tensor Output => lastLayer.output.GetCopy();

    public float ComputeError(Array ideal)
    {
        Tensor tensor = Tensor.AddBatchDimension(ideal);
        return loss.ComputeError(tensor, Output);
    }

    private Input firstLayer;
    private Output lastLayer;
    private Layer tempLayer;
    private Loss loss;
    private int helpIndex = 0;

    public void Add(Layer layer)
    {
        if (helpIndex == 0)
        {
            if (layer is not Input input) throw new Exception();
            else firstLayer = input;
            tempLayer = layer;
            helpIndex++;
            return;
        }

        tempLayer = layer.Apply(tempLayer);
    }

    public void Init(Optimizer optimizer, Loss loss)
    {
        this.loss = loss;
        
        if (tempLayer is Output o)
        lastLayer = o;
        else lastLayer = (Output)new Output().Apply(tempLayer);

        commander = new LayerCommander();

        firstLayer.InitGraph(optimizer, commander);

        Training = true;
        Initialized = true;
    }

    public void ForwardBatch(Tensor forwardData)
    {
        firstLayer.Forward(forwardData, Training);
    }

    public Tensor Forward(Tensor forwardData)
    {
        firstLayer.Forward(forwardData, Training);
        return Output;
    }

    public Tensor Forward(Array forwardData)
    {
        firstLayer.Forward(forwardData, Training);
        return Output;
    }

    public void BackPropBatch(Tensor ideal)
    {
        Tensor deriv = loss.Derivate(ideal, lastLayer.OutputTensor);

        lastLayer.BackProp(deriv);

        //ErrorRendering?.Invoke(loss.LossValue);

        //commander.CorrectionForAll();

        iteration++;
    }

    public void BackProp(Tensor ideal)
    {
        BackPropBatch(ideal);
    }

    public void BackProp(Array ideal)
    {
        BackPropBatch(Tensor.AddBatchDimension(ideal));
    }

    public void ResetModel()
    {
        commander.ResetForAll();
    }

    public void Train(Tensor[] forwardData, Tensor[] backPropData, int epochsToTrain = 1, bool batched = false, Action<long> IterationRendering = null, Action<long> EpochRendering = null, Action<float> LossRendering = null)
    {
        Training = true;

        if (!batched) 
        {
            forwardData = Tensor.GetTrainBatches(forwardData, firstLayer.inputShape.n0);
            backPropData =  Tensor.GetTrainBatches(backPropData, firstLayer.inputShape.n0);
        }

        for (int epoch = 0; epoch < epochsToTrain; epoch++)
        {
            for (int i = 0; i < forwardData.Length; i++)
            {
                this.ForwardBatch(forwardData[i]);
                this.BackPropBatch(backPropData[i]);
                IterationRendering?.Invoke(epoch * forwardData.Length + i);
                LossRendering?.Invoke(loss.LossValue);
            }
            EpochRendering?.Invoke(epoch);
        }

        Training = false;
    }

    public void Train(Array[] forwardData, Array[] backPropData, int epochsToTrain = 1)
    {
        Tensor[] batchesForward = Tensor.GetTrainBatches(forwardData, firstLayer.inputShape.n0);
        Tensor[] batchesBackProp = Tensor.GetTrainBatches(backPropData, firstLayer.inputShape.n0);

        Train(batchesForward, batchesBackProp, epochsToTrain, batched: true);
    }

    public (double accuracy, double lossValue) OneHotTest(Tensor[] forwardData, Tensor[] idealBackPropData)
    {
        int pass = 0;

        double lossValueAverage = 0, accuracy;

        for (int i = 0; i < forwardData.Length; i++)
        {
            firstLayer.Forward(forwardData[i], false);

            if (Output.IndexOfMax() == idealBackPropData[i].IndexOfMax())
                pass++;

            lossValueAverage += loss.ComputeError(idealBackPropData[i], Output);
        }

        lossValueAverage /= forwardData.Length;
        accuracy = (double)pass / forwardData.Length;

        return (accuracy, lossValueAverage);
    }
}

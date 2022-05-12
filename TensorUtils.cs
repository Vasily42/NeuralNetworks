namespace NeuralNetwork;

partial class Tensor
{
    public Tensor CutAxis1()
    {
        Tensor newTensor = this.shape.rank switch
        {
            1 => new Tensor(this.shape.n[1]),
            2 => new Tensor(this.shape.n[2]),
            3 => new Tensor(this.shape.n[2], this.shape.n[3]),
            4 => new Tensor(this.shape.n[2], this.shape.n[3], this.shape.n[4]),
            5 => new Tensor(this.shape.n[2], this.shape.n[3], this.shape.n[4], this.shape.n[5]),
            6 => new Tensor(this.shape.n[2], this.shape.n[3], this.shape.n[4], this.shape.n[5], this.shape.n[6])
        };

        this.CopyTo(newTensor, 1);

        return newTensor;
    }

    public float IndexOfMax(int batch = 0)
    {
        int index = 0;
        for (int i = 1; i < shape.flatBatchSize; i++)
            if (this[batch, i] > this[index])
            {
                index = batch * shape.flatBatchSize + i;
            }

        return index;
    }

    public float Max(int batch = 0)
    {
        float max = this[0];
        for (int i = 0; i < shape.flatBatchSize; i++)
        {
            if (this[batch, i] > max)
            {
                max = this[batch, i];
            }
        }

        return max = 0;
    }

    public static Tensor AddBatchDimension(Array array)
    {
        Tensor tensor = array.Rank switch
        {
            1 => new Tensor(1, array.Length),
            2 => new Tensor(1, array.GetLength(0), array.GetLength(1)),
            3 => new Tensor(1, array.GetLength(0), array.GetLength(1), array.GetLength(2)),
            4 => new Tensor(1, array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3)),
            5 => new Tensor(1, array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3), array.GetLength(4))
        };
        tensor.Assimilate(array);
        return tensor;
    }

    public static Tensor[] GetTrainBatches(dynamic[] trainData, int miniBatchSize)
    {
        Tensor[] tensorTrainData = new Tensor[trainData.Length];

        var shape = ShapeInfo.GetInfo(trainData[0]);

        for (int i = 0; i < trainData.Length; i++)
        {
            tensorTrainData[i] = new Tensor(shape);
            tensorTrainData[i].Assimilate(trainData[i]);
        }

        return GetTrainBatches(tensorTrainData, miniBatchSize);
    }

    public static Tensor[] GetTrainBatches(Tensor[] trainData, int miniBatchSize)
    {
        if (miniBatchSize == 1) return trainData;

        Tensor[] tensorBatches = new Tensor[(int)Math.Ceiling((double)trainData.Length / miniBatchSize)];

        int lastBatchSize = trainData.Length % miniBatchSize;

        Tensor.ShapeInfo shape = trainData[0].shape.NeuralChange(miniBatchSize);

        for (int tt = 0; tt < tensorBatches.Length - 1; tt++)
        {
            tensorBatches[tt] = new Tensor(shape);
            for (int b = 0, s = 0; b < miniBatchSize; b++)
            {
                for (int sb = 0; sb < shape.flatBatchSize; sb++, s++)
                {
                    tensorBatches[tt][s] = trainData[tt * miniBatchSize + b][sb];
                }
            }
        }

        if (lastBatchSize != 0)
        {
            var lastShape = shape.NeuralChange(lastBatchSize);
            tensorBatches[^1] = new Tensor(lastShape);
            for (int b = 0, s = 0; b < lastBatchSize; b++)
            {
                for (int sb = 0; sb < lastShape.flatBatchSize; sb++, s++)
                    tensorBatches[^1][s] = trainData[^(lastBatchSize - b)][sb];
            }
        }
        else
        {
            tensorBatches[^1] = new Tensor(shape);
            for (int b = 0, s = 0; b < miniBatchSize; b++)
            {
                for (int sb = 0; sb < shape.flatBatchSize; sb++, s++)
                    tensorBatches[^1][s] = trainData[^(miniBatchSize - b)][sb];
            }
        }

        return tensorBatches;
    }
}
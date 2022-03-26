using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;

namespace NeuralNetwork;

public unsafe class Tensor
{
    public ShapeInfo shape;
    private GCHandle pin;
    private readonly float* ptr;

    readonly float[] array;

    public Tensor(int n1, int n2 = 0, int n3 = 0, int n4 = 0, int n5 = 0, int n6 = 0)
    {
        shape = new ShapeInfo(n1, n2, n3, n4, n5, n6);

        array = new float[shape.flatSize];

        pin = GCHandle.Alloc(array, GCHandleType.Pinned);

        ptr = (float*)pin.AddrOfPinnedObject();
    }

    public Tensor(ShapeInfo shape)
    {
        this.shape = shape;

        array = new float[shape.flatSize];

        pin = GCHandle.Alloc(array, GCHandleType.Pinned);

        ptr = (float*)pin.AddrOfPinnedObject();
    }

    ~Tensor()
    {
        pin.Free();
    }

    public static Tensor Create(dynamic array)
    {
        ShapeInfo info = ShapeInfo.GetInfo(array);
        Tensor tensor = new Tensor(info);
        tensor.Assimilate(array);
        return tensor;
    }

    public void Assimilate(Array array)
    {
        GCHandle pinLoc = GCHandle.Alloc(array, GCHandleType.Pinned);
        float* p = (float*)pinLoc.AddrOfPinnedObject();
        for (int i = 0; i < shape.flatSize; i++)
            this[i] = p[i];
        pinLoc.Free();
    }

    public void CopyTo(Tensor destination, int batches = -1)
    {
        Buffer.BlockCopy(array, 0, destination.array, 0, batches == -1? shape.flatSize * 4 : batches * shape.flatBatchSize * 4);
    }

    public Tensor CutAxis1()
    {
        Tensor newTensor = this.shape.rank switch
        {
            1 => new Tensor(this.shape.n1),
            2 => new Tensor(this.shape.n2),
            3 => new Tensor(this.shape.n2, this.shape.n3),
            4 => new Tensor(this.shape.n2, this.shape.n3, this.shape.n4),
            5 => new Tensor(this.shape.n2, this.shape.n3, this.shape.n4, this.shape.n5),
            6 => new Tensor(this.shape.n2, this.shape.n3, this.shape.n4, this.shape.n5, this.shape.n6)
        };

        this.CopyTo(newTensor, 1);

        return newTensor;
    }

    public void Reshape(ShapeInfo newShape)
    {
        if (shape.flatSize != newShape.flatSize) throw new Exception();
        shape = newShape;
    }

    public static Tensor Reshape(ShapeInfo newShape, Tensor sourceTensor) 
    {
        if (sourceTensor.shape.flatSize != newShape.flatSize) throw new Exception();

        var newTensor = new Tensor(newShape);
        sourceTensor.CopyTo(newTensor);

        return newTensor;
    }

    public Tensor Fill(float @const)
    {
        for (int i = 0; i < this.shape.flatSize; i++)
            this[i] = @const;
        return this;
    }

    public Tensor Fill(Func<float> filler)
    {
        for (int i = 0; i < this.shape.flatSize; i++)
            this[i] = filler.Invoke();
        return this;
    }

    public float IndexOfMax()
    {
        float max = this[0];
        int index = 0;
        for (int i = 1; i < shape.flatBatchSize; i++)
            if (this[i] > max)
            {
                max = this[i];
                index = i;
            }

        return index;
    }

    public static Tensor AddBatchDimension(dynamic array)
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
                    tensorBatches[tt][s] = trainData[tt][b, sb];
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

    public float this[int f]
    {
        get => ptr[f];
        set => ptr[f] = value;
    }

    public float this[int n1, int f]
    {
        get => ptr[n1 * shape.n26 + f];
        set => ptr[n1 * shape.n26 + f] = value;
    }

    public float this[int n1, int n2, int f]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + f];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + f] = value;
    }

    public float this[int n1, int n2, int n3, int f]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + f];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + f] = value;
    }

    public float this[int n1, int n2, int n3, int n4, int f]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + f];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + f] = value;
    }

    public float this[int n1, int n2, int n3, int n4, int n5, int n6]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + n5 * shape.n6 + n6];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + n5 * shape.n6 + n6] = value;
    }

    public float this[ShapeInfo shape, int n1, int f]
    {
        get => ptr[n1 * shape.n26 + f];
        set => ptr[n1 * shape.n26 + f] = value;
    }

    public float this[ShapeInfo shape, int n1, int n2, int f]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + f];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + f] = value;
    }

    public float this[ShapeInfo shape, int n1, int n2, int n3, int f]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + f];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + f] = value;
    }

    public float this[ShapeInfo shape, int n1, int n2, int n3, int n4, int f]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + f];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + f] = value;
    }

    public float this[ShapeInfo shape, int n1, int n2, int n3, int n4, int n5, int n6]
    {
        get => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + n5 * shape.n6 + n6];
        set => ptr[n1 * shape.n26 + n2 * shape.n36 + n3 * shape.n46 + n4 * shape.n56 + n5 * shape.n6 + n6] = value;
    }

    public readonly struct ShapeInfo
    {
        public readonly int n1, n2, n3, n4, n5, n6, n26, n36, n46, n56,
        xLength, yLength, zLength, wLength,
        channels, batchSize, flatBatchSize, flatSize;

        public readonly byte rank;

        public ShapeInfo(int n1, int n2 = 0, int n3 = 0, int n4 = 0, int n5 = 0, int n6 = 0)
        {
            this.n1 = n1;
            this.n2 = n2;
            this.n3 = n3;
            this.n4 = n4;
            this.n5 = n5;
            this.n6 = n6;

            this.rank = 1;

            if (n2 > 0) rank++;
            else n2 = 1;
            if (n3 > 0) rank++;
            else n3 = 1;
            if (n4 > 0) rank++;
            else n4 = 1;
            if (n5 > 0) rank++;
            else n5 = 1;
            if (n6 > 0) rank++;
            else n6 = 1;

            this.flatSize = n1 * n2 * n3 * n4 * n5 * n6;

            n26 = flatSize / n1;
            n36 = n26 / n2;
            n46 = n36 / n3;
            n56 = n46 / n4;

            this.batchSize = 1;
            this.channels = 0;
            this.xLength = 0;
            this.yLength = 0;
            this.zLength = 0;
            this.wLength = 0;

            switch (rank)
            {
                case 1:
                    xLength = n1;
                    break;

                case 2:
                    batchSize = n1;
                    xLength = n2;
                    break;

                case 3:
                    batchSize = n1;
                    channels = n2;
                    xLength = n3;
                    break;

                case 4:
                    batchSize = n1;
                    channels = n2;
                    yLength = n3;
                    xLength = n4;
                    break;

                case 5:
                    batchSize = n1;
                    channels = n2;
                    zLength = n3;
                    yLength = n4;
                    xLength = n5;
                    break;

                case 6:
                    batchSize = n1;
                    channels = n2;
                    wLength = n3;
                    zLength = n4;
                    yLength = n5;
                    xLength = n6;
                    break;
            }

            flatBatchSize = flatSize / batchSize;

            if(rank == 1) batchSize = 0;
        }

        public static ShapeInfo NeuralCreate(byte rank, int xLength, int batchSize = 0, int channels = 0, int yLength = 0, int zLength = 0, int wLength = 0)
        {        
            return rank switch
            {
                1 => new(xLength),
                2 => new(batchSize, xLength),
                3 => new(batchSize, channels, xLength),
                4 => new(batchSize, channels, yLength, xLength),
                5 => new(batchSize, channels, zLength, yLength, xLength),
                6 => new(batchSize, channels, wLength, zLength, yLength, xLength)
            };
        }

        public ShapeInfo NeuralChange(int batchSize = -1, int channels = -1,
          int wLength = -1, int zLength = -1, int yLength = -1, int xLength = -1)
        {
            return NeuralCreate(this.rank,
            xLength != -1 ? xLength : this.xLength,
            batchSize != -1 ? batchSize : this.batchSize,
            channels != -1 ? channels : this.channels,
            yLength != -1 ? yLength : this.yLength,
            zLength != -1 ? zLength : this.zLength,
            wLength != -1 ? wLength : this.wLength);
        }

        public static ShapeInfo GetInfo(Array array) => array.Rank switch
        {
            1 => new(array.Length),
            2 => new(array.GetLength(0), array.GetLength(1)),
            3 => new(array.GetLength(0), array.GetLength(1), array.GetLength(2)),
            4 => new(array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3)),
            5 => new(array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3), array.GetLength(4)),
            6 => new(array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3), array.GetLength(4), array.GetLength(5)),
            _ => throw new Exception(),
        };

        public static implicit operator ShapeInfo(int x)
        => new(x);
        public static implicit operator ShapeInfo((int n1, int n2) tupl)
        => new(tupl.n1, tupl.n2);
        public static implicit operator ShapeInfo((int n1, int n2, int n3) tupl)
        => new(tupl.n1, tupl.n2, tupl.n3);
        public static implicit operator ShapeInfo((int n1, int n2, int n3, int n4) tupl)
        => new(tupl.n1, tupl.n2, tupl.n3, tupl.n4);
        public static implicit operator ShapeInfo((int n1, int n2, int n3, int n4, int n5) tupl)
        => new(tupl.n1, tupl.n2, tupl.n3, tupl.n4, tupl.n5);
        public static implicit operator ShapeInfo((int n1, int n2, int n3, int n4, int n5, int n6) tupl)
        => new(tupl.n1, tupl.n2, tupl.n3, tupl.n4, tupl.n5, tupl.n6);
    }
}

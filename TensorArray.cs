using System.Runtime.InteropServices;
using System.Runtime.Serialization;

namespace NeuralNetwork;

public unsafe abstract class Tensor
{
    public readonly ShapeInfo shape;
    protected GCHandle pin;
    protected float* ptr;

    protected Tensor(int n1, int n2 = 0, int n3 = 0, int n4 = 0, int n5 = 0, int n6 = 0) => shape = new ShapeInfo(n1, n2, n3, n4, n5, n6);

    protected Tensor(ShapeInfo info) => shape = info;

    ~Tensor()
    {
        pin.Free();
    }

    protected void PinAndRef(object array)
    {
        pin = GCHandle.Alloc(array, GCHandleType.Pinned);

        ptr = (float*)pin.AddrOfPinnedObject();
    }

    public static Tensor CreateWithRef(Array array) => array.Rank switch
    {
        1 => new Tensor1((float[]) array),
        2 => new Tensor2((float[,])array),
        3 => new Tensor3((float[,,])array),
        4 => new Tensor4((float[,,,])array),
        5 => new Tensor5((float[,,,,])array),
        6 => new Tensor6((float[,,,,,])array),
        _ => throw new Exception(),
    };

    public static Tensor Create(ShapeInfo shape) => shape.rank switch
    {
        1 => new Tensor1(shape.n1),
        2 => new Tensor2(shape.n1, shape.n2),
        3 => new Tensor3(shape.n1, shape.n2, shape.n3),
        4 => new Tensor4(shape.n1, shape.n2, shape.n3, shape.n4),
        5 => new Tensor5(shape.n1, shape.n2, shape.n3, shape.n4, shape.n5),
        6 => new Tensor6(shape.n1, shape.n2, shape.n3, shape.n4, shape.n5, shape.n6)
    };

    public static Tensor Create(dynamic array)
    {
        ShapeInfo info = ShapeInfo.GetInfo(array);
        Tensor tensor = Tensor.Create(info);
        tensor.Assimilate(array);
        return tensor;
    }

    public void Assimilate(float[] array)
    {
        fixed (float* p = array)
            for (int i = 0; i < shape.flatSize; i++)
                this[i] = p[i];
    }

    public void Assimilate(float[,] array)
    {
        fixed (float* p = array)
            for (int i = 0; i < shape.flatSize; i++)
                this[i] = p[i];
    }

    public void Assimilate(float[,,] array)
    {
        fixed (float* p = array)
            for (int i = 0; i < shape.flatSize; i++)
                this[i] = p[i];
    }

    public void Assimilate(float[,,,] array)
    {
        fixed (float* p = array)
            for (int i = 0; i < shape.flatSize; i++)
                this[i] = p[i];
    }

    public void Assimilate(float[,,,,] array)
    {
        fixed (float* p = array)
            for (int i = 0; i < shape.flatSize; i++)
                this[i] = p[i];
    }

    public void Assimilate(float[,,,,,] array)
    {
        fixed (float* p = array)
            for (int i = 0; i < shape.flatSize; i++)
                this[i] = p[i];
    }

    public void CopyTo(Tensor destination, int batches = -1)
    {
        if (batches == -1)
            for (int i = 0; i < this.shape.flatSize; i++)
                destination[i] = this[i];
        else
            for (int batch = 0; batch < batches; batch++)
                for (int i = 0; i < shape.flatBatchSize; i++)
                    destination[batch, i] = this[batch, i];
    }

    public static void Copy(Tensor source, Tensor destination, int batches = -1)
    {
        source.CopyTo(destination, batches);
    }

    public Tensor CutAxis1()
    {
        Tensor newTensor = this.shape.rank switch
        {
            1 => new Tensor1(this.shape.n1),
            2 => new Tensor1(this.shape.n2),
            3 => new Tensor2(this.shape.n2, this.shape.n3),
            4 => new Tensor3(this.shape.n2, this.shape.n3, this.shape.n4),
            5 => new Tensor4(this.shape.n2, this.shape.n3, this.shape.n4, this.shape.n5),
            6 => new Tensor5(this.shape.n2, this.shape.n3, this.shape.n4, this.shape.n5, this.shape.n6)
        };

        this.CopyTo(newTensor, 1);

        return newTensor;
    }

    public Tensor Reshape(ShapeInfo newShape)
    {
        if (shape.flatSize != newShape.flatSize) throw new Exception();

        Tensor newTensor = Create(newShape);

        this.CopyTo(newTensor);

        return newTensor;
    }

    public static Tensor Reshape(ShapeInfo newShape, Tensor sourceTensor) => sourceTensor.Reshape(newShape);

    public Tensor Fill(float @const)
    {
        for (int i = 0; i < this.shape.flatSize; i++)
            this[i] = @const;
        return this;
    }

    public Tensor Fill(Func<float> random)
    {
        for (int i = 0; i < this.shape.flatSize; i++)
            this[i] = random.Invoke();
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
            1 => new Tensor2(1, array.Length),
            2 => new Tensor3(1, array.GetLength(0), array.GetLength(1)),
            3 => new Tensor4(1, array.GetLength(0), array.GetLength(1), array.GetLength(2)),
            4 => new Tensor5(1, array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3)),
            5 => new Tensor6(1, array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3), array.GetLength(4))
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
            tensorTrainData[i] = Tensor.Create(shape);
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
            tensorBatches[tt] = Tensor.Create(shape);
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
            tensorBatches[^1] = Tensor.Create(lastShape);
            for (int b = 0, s = 0; b < lastBatchSize; b++)
            {
                for (int sb = 0; sb < lastShape.flatBatchSize; sb++, s++)
                    tensorBatches[^1][s] = trainData[^(lastBatchSize - b)][sb];
            }
        }
        else
        {
            tensorBatches[^1] = Tensor.Create(shape);
            for (int b = 0, s = 0; b < miniBatchSize; b++)
            {
                for (int sb = 0; sb < shape.flatBatchSize; sb++, s++)
                    tensorBatches[^1][s] = trainData[^(miniBatchSize - b)][sb];
            }
        }

        return tensorBatches;
    }

    public float this[int flatIndex]
    {
        get => ptr[flatIndex];
        set => ptr[flatIndex] = value;
    }

    public float this[int n1, int flat]
    {
        get => ptr[n1 * shape.flatBatchSize + flat];
        set => ptr[n1 * shape.flatBatchSize + flat] = value;
    }

    public virtual float this[int n1, int n2, int n3]
    {
        get => -1;
        set { }
    }

    public virtual float this[int n1, int n2, int n3, int n4]
    {
        get => -1;
        set { }
    }

    public virtual float this[int n1, int n2, int n3, int n4, int n5]
    {
        get => -1;
        set { }
    }

    public virtual float this[int n1, int n2, int n3, int n4, int n5, int n6]
    {
        get => -1;
        set { }
    }

    public readonly struct ShapeInfo
    {
        public readonly int n1, n2, n3, n4, n5, n6,
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

            if (rank == 1) batchSize = 0;
        }

        public static ShapeInfo NeuralCreate(int xLength, int batchSize = 0, int channels = 0, int yLength = 0, int zLength = 0, int wLength = 0)
        {
            byte rank = 0;

            if (wLength > 0) rank++;
            else if (zLength > 0) rank++;
            else if (yLength > 0) rank++;
            else if (channels > 0) rank++;
            else if (batchSize > 0) rank++;
            else if (xLength > 0) rank++;

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
            return NeuralCreate((xLength != -1? xLength : this.xLength),
            (batchSize != -1? batchSize : this.batchSize),
            (channels != -1? channels : this.channels),
            (yLength != -1? yLength : this.yLength),
            (zLength != -1? zLength : this.zLength),
            (wLength != -1? wLength : this.wLength));
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

    public abstract Array ToArray();
}

public unsafe sealed class Tensor1 : Tensor
{
    readonly float[] array1;

    public Tensor1(int n1) : base(n1)
    {
        array1 = new float[n1];

        PinAndRef(array1);
    }

    internal Tensor1(float[] array) : base(ShapeInfo.GetInfo(array)) => array1 = array;

    public sealed override Array ToArray() => array1;
} 

public unsafe sealed class Tensor2 : Tensor
{
    readonly float[,] array2;

    public Tensor2(int n1, int n2) : base(n1, n2)
    {
        array2 = new float[n1, n2];

        PinAndRef(array2);
    }

    internal Tensor2(float[,] array) : base(ShapeInfo.GetInfo(array)) => array2 = array;

    public sealed override Array ToArray() => array2;
}

public unsafe sealed class Tensor3 : Tensor
{
    readonly float[,,] array3;

    public Tensor3(int n1, int n2, int n3) : base(n1, n2, n3)
    {
        array3 = new float[n1, n2, n3];

        PinAndRef(array3);
    }

    internal Tensor3(float[,,] array) : base(ShapeInfo.GetInfo(array)) => array3 = array;

    public sealed override float this[int n1, int n2, int n3]
    {
        get => array3[n1, n2, n3];
        set => array3[n1, n2, n3] = value;
    }

    public sealed override Array ToArray() => array3;
}

public unsafe sealed class Tensor4 : Tensor
{
    readonly float[,,,] array4;

    public Tensor4(int n1, int n2, int n3, int n4) : base(n1, n2, n3, n4)
    {
        array4 = new float[n1, n2, n3, n4];

        PinAndRef(array4);
    }

    internal Tensor4(float[,,,] array) : base(ShapeInfo.GetInfo(array)) => array4 = array;

    public sealed override float this[int n1, int n2, int n3, int n4]
    {
        get => array4[n1, n2, n3, n4];
        set => array4[n1, n2, n3, n4] = value;
    }

    public sealed override Array ToArray() => array4;
}

public unsafe sealed class Tensor5 : Tensor
{
    readonly float[,,,,] array5;

    public Tensor5(int n1, int n2, int n3, int n4, int n5) : base(n1, n2, n3, n4, n5)
    {
        array5 = new float[n1, n2, n3, n4, n5];

        PinAndRef(array5);
    }

    internal Tensor5(float[,,,,] array) : base(ShapeInfo.GetInfo(array)) => array5 = array;

    public sealed override float this[int n1, int n2, int n3, int n4, int n5]
    {
        get => array5[n1, n2, n3, n4, n5];
        set => array5[n1, n2, n3, n4, n5] = value;
    }

    public sealed override Array ToArray() => array5;
}

public unsafe sealed class Tensor6 : Tensor
{
    readonly float[,,,,,] array6;

    public Tensor6(int n1, int n2, int n3, int n4, int n5, int n6) : base(n1, n2, n3, n4, n5, n6)
    {
        array6 = new float[n1, n2, n3, n4, n5, n6];

        PinAndRef(array6);
    }

    internal Tensor6(float[,,,,,] array) : base(ShapeInfo.GetInfo(array)) => array6 = array;

    public sealed override float this[int n1, int n2, int n3, int n4, int n5, int n6]
    {
        get => array6[n1, n2, n3, n4, n5, n6];
        set => array6[n1, n2, n3, n4, n5, n6] = value;
    }

    public sealed override Array ToArray() => array6;
}

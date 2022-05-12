using System.Runtime.InteropServices;

namespace NeuralNetwork;

public unsafe partial class Tensor
{
    public ShapeInfo shape;
    private GCHandle pin;
    private readonly float* ptr;

    internal readonly float[] array;

    public Tensor(params int[] n)
    {
        shape = new ShapeInfo(n);

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

    public static Tensor Create(Array array)
    {
        ShapeInfo info = ShapeInfo.GetInfo(array);
        Tensor tensor = new(info);
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
        Buffer.BlockCopy(array, 0, destination.array, 0, batches == -1 ? shape.flatSize * 4 : batches * shape.flatBatchSize * 4);
        // for (int i = 0; i < shape.flatSize; i++) destination[i] = this[i];
    }

    public Tensor GetCopy(int batches = -1)
    {
        Tensor newTensor = new(this.shape);
        this.CopyTo(newTensor, batches);
        return newTensor;
    }

    public Tensor Reshape(ShapeInfo newShape)
    {
        if (shape.flatSize != newShape.flatSize) throw new Exception();
        shape = newShape;
        return this;
    }

    public static Tensor Reshape(ShapeInfo newShape, Tensor sourceTensor)
    {
        if (sourceTensor.shape.flatSize != newShape.flatSize) throw new Exception();

        var newTensor = new Tensor(newShape);
        sourceTensor.CopyTo(newTensor);

        return newTensor;
    }

    public static unsafe Tensor Concat(Tensor[] fragments, int axis = -1)
    {
        if (axis < 0) axis = fragments[0].shape.rank + axis;

        int[] newSh = new int[fragments[0].shape.rank];

        fragments[0].shape.n.CopyTo(newSh, 0);

        for (int i = 1; i < fragments.Length; i++)
        {
            newSh[axis] += fragments[i].shape.n[axis];
        }

        Tensor catTensor = new(newSh);

        int nFRev = fragments[0].shape.flatSize / fragments[0].shape.nF[axis];

        for (int i = 0; i < nFRev; i++)
            for (int j = 0, k = 0; j < fragments.Length; j++)
                for (int q = 0; q < fragments[j].shape.nF[axis]; q++, k++) catTensor[i * catTensor.shape.nF[axis] + k] = fragments[j][i * fragments[j].shape.nF[axis] + q];

        return catTensor;
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

    public float this[int f]
    {
        get => ptr[f];
        set => ptr[f] = value;
    }

    public float this[int n1, int f]
    {
        get 
        {
            #if DEBUG
            if (n1 * shape.nF1 + f >= shape.flatSize) throw new Exception();
            #endif
            return ptr[n1 * shape.nF1 + f];
        }
        set 
        {
            #if DEBUG
            if (n1 * shape.nF1 + f >= shape.flatSize) throw new Exception();
            #endif
            ptr[n1 * shape.nF1 + f] = value;
        }
    }

    public float this[int n1, int n2, int f]
    {
        get 
        {
            #if DEBUG
            if (n1 * shape.nF1 + n2 * shape.nF2 + f >= shape.flatSize) throw new Exception();
            #endif
            return ptr[n1 * shape.nF1 + n2 * shape.nF2 + f];
        }

        set 
        {
            #if DEBUG
            if (n1 * shape.nF1 + n2 * shape.nF2 + f >= shape.flatSize) throw new Exception();
            #endif
            ptr[n1 * shape.nF1 + n2 * shape.nF2 + f] = value;
        }
    }

    public float this[int n1, int n2, int n3, int f]
    {
        get 
        {
            #if DEBUG
            if (n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + f >= shape.flatSize) throw new Exception();
            #endif
            return ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + f];
        }
        set
        {
            #if DEBUG
            if (n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + f >= shape.flatSize) throw new Exception();
            #endif
            ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + f] = value;
        }
    }

    public float this[int n1, int n2, int n3, int n4, int f]
    {
        get => ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + n4 * shape.nF4 + f];
        set => ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + n4 * shape.nF4 + f] = value;
    }

    public float this[int n1, int n2, int n3, int n4, int n5, int f]
    {
        get => ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + n4 * shape.nF4 + n5 * shape.nF5 + f];
        set => ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + n4 * shape.nF4 + n5 * shape.nF5 + f] = value;
    }

    public float this[params int[] n]
    {
        get
        {
            int ind = 0;
            for (int i = 1; i < n.Length - 1; i++) ind += shape.nF[i] * n[i];
            return ptr[ind + n[^1]];
        }

        set
        {
            int ind = 0;
            for (int i = 1; i < n.Length - 1; i++) ind += shape.nF[i] * n[i];
            ptr[ind + n[^1]] = value;
        }
    }

    public static Tensor operator /(Tensor a, float div)
    {
        for (int i = 0; i < a.shape.flatSize; i++) a[i] /= div;
        return a;
    }

    public sealed override string ToString()
    {
        return String.Join(" ", array);
    }
}

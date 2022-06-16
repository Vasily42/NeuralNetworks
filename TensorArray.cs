global using System.Numerics;
global using System.Runtime.InteropServices;
global using System.Runtime.CompilerServices;
using System.Globalization;
using System.Text;
using System.Drawing;
using System.ComponentModel;

namespace NeuralNetwork;

[Serializable]
public unsafe partial class Tensor
{
    public static readonly int vecCount;

    static Tensor()
    {
        vecCount = Vector<float>.Count;
    }
    
    public ShapeInfo shape;
    private readonly float* ptr;

    private readonly bool isSlice = false;

    public ReadOnlySpan<float> Span
    {
        get => new(ptr, shape);
    }

    public Tensor(params int[] n)
    {
        shape = new ShapeInfo(n);

        ptr = (float*)NativeMemory.AlignedAlloc((nuint)shape.nF0 * 4, (nuint)sizeof(nuint));
    }

    public Tensor(ShapeInfo shape)
    {
        this.shape = shape;

        ptr = (float*)NativeMemory.AlignedAlloc((nuint)shape.nF0 * 4, (nuint)sizeof(nuint));
    }

    private Tensor(float* ptr, ShapeInfo shape)
    {
        this.shape = shape;
        this.ptr = ptr;
        isSlice = true;
    }

    ~Tensor()
    {
        if (!isSlice)
        NativeMemory.AlignedFree(ptr);
    }

    public static Tensor Create(Array array)
    {
        ShapeInfo info = ShapeInfo.GetInfo(array);
        Tensor tensor = new(info);
        tensor.Assimilate(array);
        return tensor;
    }

    public Tensor Slice(int startIndex, int length)
    {
        return new(ptr + startIndex, new(length));
    }

    public void Assimilate(Array array)
    {
        void* ptrArr = (void*)GCHandle.Alloc(array, GCHandleType.Pinned).AddrOfPinnedObject();

        Buffer.MemoryCopy(ptrArr, ptr, shape * 4, array.Length * 4);
    }

    public void CopyTo(Tensor destination)
    {
        Buffer.MemoryCopy(ptr, destination.ptr, shape * 4, shape * 4);
    }

    public void CopyTo(Tensor destination, int batches)
    {
        Buffer.MemoryCopy(ptr, destination.ptr, shape * 4, batches * shape.nF1 * 4);
    }

    public Tensor GetCopy()
    {
        Tensor newTensor = new(this.shape);
        this.CopyTo(newTensor);
        return newTensor;
    }

    public Tensor GetCopy(int batches)
    {
        Tensor newTensor = new(this.shape);
        this.CopyTo(newTensor, batches);
        return newTensor;
    }

    public Tensor Reshape(ShapeInfo newShape)
    {
        if (shape.nF0 != newShape.nF0) throw new Exception();
        shape = newShape;
        return this;
    }

    public static Tensor Reshape(ShapeInfo newShape, Tensor sourceTensor)
    {
        if (sourceTensor.shape.nF0 != newShape.nF0) throw new Exception();

        var newTensor = new Tensor(newShape);
        sourceTensor.CopyTo(newTensor);

        return newTensor;
    }

    public static unsafe Tensor Concat(Tensor[] fragments, int axis = -1)
    {
        if (axis < 0) axis = fragments[0].shape.rank + axis;

        int[] newSh = new int[fragments[0].shape.rank];

        fragments[0].shape.N.CopyTo(newSh, 0);

        for (int i = 1; i < fragments.Length; i++)
        {
            newSh[axis] += fragments[i].shape[axis];
        }

        Tensor catTensor = new(newSh);

        int nFRev = fragments[0].shape.nF0 / fragments[0].shape.NF[axis];

        for (int i = 0; i < nFRev; i++)
            for (int j = 0, k = 0; j < fragments.Length; j++)
                for (int q = 0; q < fragments[j].shape.NF[axis]; q++, k++) catTensor[i * catTensor.shape.NF[axis] + k] = fragments[j][i * fragments[j].shape.NF[axis] + q];

        return catTensor;
    }

    public Tensor Clear()
    {
        Unsafe.InitBlock(ptr, 0, (uint)(shape * 4));
        return this;
    }

    public Tensor Clear(int startIndex, int length)
    {
        Unsafe.InitBlock(ptr + startIndex, 0, (uint)(length * 4));
        return this;
    }

    public Tensor Fill(float @const)
    {
        for (int i = 0; i < shape; i++) this[i] = @const;
        return this;
    }

    public Tensor Fill(float @const, int startIndex, int length)
    {
        for (int i = startIndex; i < startIndex + length; i++) this[i] = @const;
        return this;
    }

    public Tensor Fill(Func<float> filler)
    {
        for (int i = 0; i < this.shape; i++) this[i] = filler.Invoke();
        return this;
    }

    public float this[int f]
    {
        get
        {
            return ptr[f];
        }
        set
        {
            ptr[f] = value;
        }
    }

    public float this[int n1, int f]
    {
        get
        {
            return ptr[n1 * shape.nF1 + f];
        }
        set
        {
            ptr[n1 * shape.nF1 + f] = value;
        }
    }

    public float this[int n1, int n2, int f]
    {
        get
        {
            return ptr[n1 * shape.nF1 + n2 * shape.nF2 + f];
        }

        set
        {
            ptr[n1 * shape.nF1 + n2 * shape.nF2 + f] = value;
        }
    }

    public float this[int n1, int n2, int n3, int f]
    {
        get
        {
            return ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + f];
        }
        set
        {
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

    public float this[int n1, int n2, int n3, int n4, int n5, int n6, int f]
    {
        get => ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + n4 * shape.nF4 + n5 * shape.nF5 + n6 * shape.nF6 + f];
        set => ptr[n1 * shape.nF1 + n2 * shape.nF2 + n3 * shape.nF3 + n4 * shape.nF4 + n5 * shape.nF5 + n6 * shape.nF6 + f] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<float> GetVector(int index)
    {
        return new Vector<float>(new ReadOnlySpan<float>(ptr + index, vecCount));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void InsertVector(ref Vector<float> a, int index)
    {
        a.CopyTo(new Span<float>(ptr + index, vecCount));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<Vector<float>> GetSpanVectors(out int remain)
    {
        var res = MemoryMarshal.Cast<float, Vector<float>>(new Span<float>(ptr, shape));
        remain = res.Length * vecCount;
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<Vector<float>> GetSpanVectors()
    {
        return MemoryMarshal.Cast<float, Vector<float>>(new Span<float>(ptr, shape));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<Vector<float>> GetSpanVectors(out int remain, int startIndex, int len)
    {
        Span<Vector<float>> res = MemoryMarshal.Cast<float, Vector<float>>(new Span<float>(ptr + startIndex, len));
        remain = res.Length * vecCount;
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<Vector<float>> GetSpanVectors(int startIndex, int len)
    {
        return MemoryMarshal.Cast<float, Vector<float>>(new Span<float>(ptr + startIndex, len));
    }

    public sealed override string ToString()
    {
        StringBuilder sb = new();
        for (int i = 0; i < shape - 1; i++)
        {
            sb.Append(this[i]);
            sb.Append(' ');
        }
        sb.Append(this[shape - 1]);

        return sb.ToString();
    }
}

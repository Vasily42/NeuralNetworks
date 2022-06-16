using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace NeuralNetwork;

partial class Tensor
{
    private const float eHelpApprox = 0.03125f;

    public static Tensor operator +(Tensor a, float term)
    {
        int remain;
        Tensor b = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = new Vector<float>(term);
        for (int i = 0; i < aVec.Length; i++) bVec[i] = aVec[i] + cVec;
        for (; remain < a.shape; remain++) b[remain] = a[remain] + term;
        return b;
    }

    public static Tensor operator -(Tensor a, float sub)
    {
        int remain;
        Tensor b = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = new Vector<float>(sub);
        for (int i = 0; i < aVec.Length; i++) bVec[i] = aVec[i] / cVec;
        for (; remain < a.shape; remain++) b[remain] = a[remain] / sub;
        return b;
    }

    public static Tensor operator *(Tensor a, float mult)
    {
        int remain;
        Tensor b = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = new Vector<float>(mult);
        for (int i = 0; i < aVec.Length; i++) bVec[i] = aVec[i] / cVec;
        for (; remain < a.shape; remain++) b[remain] = a[remain] / mult;
        return b;
    }

    public static Tensor operator /(Tensor a, float div)
    {
        int remain;
        Tensor b = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = new Vector<float>(div);
        for (int i = 0; i < aVec.Length; i++) bVec[i] = aVec[i] / cVec;
        for (; remain < a.shape; remain++) b[remain] = a[remain] / div;
        return b;
    }

    public static Tensor operator +(Tensor a, Tensor b)
    {
        int remain;
        Tensor c = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = c.GetSpanVectors();
        for (int i = 0; i < aVec.Length; i++) cVec[i] = aVec[i] + bVec[i];
        for (; remain < a.shape; remain++) c[remain] = a[remain] + b[remain];
        return c;
    }

    public static Tensor operator -(Tensor a, Tensor b)
    {
        int remain;
        Tensor c = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = c.GetSpanVectors();
        for (int i = 0; i < aVec.Length; i++) cVec[i] = aVec[i] - bVec[i];
        for (; remain < a.shape; remain++) c[remain] = a[remain] - b[remain];
        return c;
    }

    public static Tensor operator *(Tensor a, Tensor b)
    {
        int remain;
        Tensor c = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = c.GetSpanVectors();
        for (int i = 0; i < aVec.Length; i++) cVec[i] = aVec[i] * bVec[i];
        for (; remain < a.shape; remain++) c[remain] = a[remain] * b[remain];
        return c;
    }

    public static Tensor operator /(Tensor a, Tensor b)
    {
        int remain;
        Tensor c = new(a.shape);
        var aVec = a.GetSpanVectors(out remain);
        var bVec = b.GetSpanVectors();
        var cVec = c.GetSpanVectors();
        for (int i = 0; i < aVec.Length; i++) cVec[i] = aVec[i] / bVec[i];
        for (; remain < a.shape; remain++) c[remain] = a[remain] / b[remain];
        return c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static public float Dot(Tensor a, Tensor b, int startIndexA, int startIndexB, int len)
    {
        int remainA, remainB;

        var accum = Vector<float>.Zero;

        float result = 0;

        var aVec = a.GetSpanVectors(out remainA, startIndexA, len);
        var bVec = b.GetSpanVectors(out remainB, startIndexB, len);

        for (int i = 0; i < aVec.Length; i++) accum += aVec[i] * bVec[i];      

        result += Vector.Sum<float>(accum);

        for (int i = 0; i < len % Tensor.vecCount; i++) result += a[remainA + i] * b[remainB + i];

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe public Vector<float> Exp(Vector<float> a)
    {
        float* buffer = stackalloc float[vecCount];

        for (int i = 0; i < vecCount; i++)
        {
            buffer[i] = MathF.Exp(a[i]);
        }

        var res = new Vector<float>(new ReadOnlySpan<float>(buffer, vecCount));

        // var c = Vector.GreaterThan(a, Vector<float>.Zero);

        // var aSh = -Vector.Abs(a);

        // var n = new Vector<float>(eHelpApprox);

        // var res = Vector<float>.One + (aSh * n);

        // res = res * res * res * res * res;

        // res = Vector.ConditionalSelect(c, Vector<float>.One / res, res);

        return res;
    }
}
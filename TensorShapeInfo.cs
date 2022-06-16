namespace NeuralNetwork;

partial class Tensor
{
    [Serializable]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe readonly struct ShapeInfo
    {
        [FieldOffset(0)] public readonly int n0 = 0;
        [FieldOffset(4)] public readonly int n1 = 0;
        [FieldOffset(8)] public readonly int n2 = 0;
        [FieldOffset(12)] public readonly int n3 = 0;
        [FieldOffset(16)] public readonly int n4 = 0;
        [FieldOffset(20)] public readonly int n5 = 0;
        [FieldOffset(24)] public readonly int n6 = 0;
        [FieldOffset(28)] public readonly int nF0 = 0;
        [FieldOffset(32)] public readonly int nF1 = 0;
        [FieldOffset(36)] public readonly int nF2 = 0;
        [FieldOffset(40)] public readonly int nF3 = 0;
        [FieldOffset(44)] public readonly int nF4 = 0;
        [FieldOffset(48)] public readonly int nF5 = 0;
        [FieldOffset(52)] public readonly int nF6 = 0;
        [FieldOffset(56)] public readonly byte rank = 0;

        public int[] N
        {
            get 
            {
                int[] n = new int[7];

                int* pArr = (int*)GCHandle.Alloc(n, GCHandleType.Pinned).AddrOfPinnedObject();

                fixed (ShapeInfo* pt = &this)
                {
                    int* p = (int*)pt;

                    Unsafe.CopyBlock(pArr, p, (uint)28);
                }

                return n;
            }
        }

        public int[] NF
        {
            get 
            {
                int[] nF = new int[7];

                int* pArr = (int*)GCHandle.Alloc(nF, GCHandleType.Pinned).AddrOfPinnedObject();

                fixed (ShapeInfo* pt = &this)
                {
                    int* p = ((int*)pt) + 7;

                    Unsafe.CopyBlock(pArr, p, (uint)28);
                }

                return nF;
            }
        }

        public ShapeInfo(params int[] n)
        {
            this.rank = (byte)n.Length;

            if (rank > 7) throw new ArgumentException($"too many dimensions ({rank})");

            fixed (ShapeInfo* pt = &this)
            {
                int* p = (int*)pt;
                p[6 + rank] = n[^1];
                p[rank - 1] = n[^1];
                for (int i = rank - 2; i >= 0; i--)
                {
                    p[i] = n[i];
                    p[7 + i] = p[8 + i] * p[i];
                }
            }
        }

        public ShapeInfo Change(params (int axis, int newSize)[] mods)
        {
            int[] newSh = new int[rank];

            fixed (ShapeInfo* pt = &this)
            {
                int* p = (int*)pt;
                for (int i = 0; i < rank; i++) newSh[i] = p[i];
            }

            for (int i = 0; i < mods.Length; i++) newSh[mods[i].axis] = mods[i].newSize;
            return new ShapeInfo(newSh);
        }

        public static ShapeInfo GetInfo(Array array)
        {
            int[] n = new int[array.Rank];

            for (int i = 0; i < array.Rank; i++) n[i] = array.GetLength(i);

            return new ShapeInfo(n);
        }

        public int this[int ax]
        {
            get
            {
                fixed (ShapeInfo* pt = &this)
                {
                    int * p = (int*)pt;
                    return p[ax];
                }
            }
        }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Point(int n0, int f) => n0 * nF1 + f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Point(int n0, int n1, int f) => n0 * nF1 + n1 * nF2 + f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Point(int n0, int n1, int n2, int f) => n0 * nF1 + n1 * nF2 + n2 * nF3 + f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Point(int n0, int n1, int n2, int n3, int f) => n0 * nF1 + n1 * nF2 + n2 * nF3 + n3 * nF4 + f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Point(int n0, int n1, int n2, int n3, int n4, int f) => n0 * nF1 + n1 * nF2 + n2 * nF3 + n3 * nF4 + n4 * nF5 + f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Point(int n0, int n1, int n2, int n3, int n4, int n5, int f) => n0 * nF1 + n1 * nF2 + n2 * nF3 + n3 * nF4 + n4 * nF5 + n5 * nF6 + f;

        public static implicit operator Tensor(ShapeInfo a)
        => new Tensor(a).Fill(0);
        public static implicit operator int(ShapeInfo a)
        => a.nF0;
        public static implicit operator ShapeInfo(int n0)
        => new(n0);
        public static implicit operator ShapeInfo((int n0, int n1) tupl)
        => new(tupl.n0, tupl.n1);
        public static implicit operator ShapeInfo((int n0, int n1, int n2) tupl)
        => new(tupl.n0, tupl.n1, tupl.n2);
        public static implicit operator ShapeInfo((int n0, int n1, int n2, int n3) tupl)
        => new(tupl.n0, tupl.n1, tupl.n2, tupl.n3);
        public static implicit operator ShapeInfo((int n0, int n1, int n2, int n3, int n4) tupl)
        => new(tupl.n0, tupl.n1, tupl.n2, tupl.n3, tupl.n4);
        public static implicit operator ShapeInfo((int n0, int n1, int n2, int n3, int n4, int n5) tupl)
        => new(tupl.n0, tupl.n1, tupl.n2, tupl.n3, tupl.n4, tupl.n5);
    }
}
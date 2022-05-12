namespace NeuralNetwork;

partial class Tensor
{
    public readonly struct ShapeInfo
    {
        public readonly int n0, n1, n2, n3, n4, n5;
        public readonly int nF0, nF1, nF2, nF3, nF4, nF5;
        public readonly byte rank;
        public readonly int xLength, yLength, zLength, wLength,
        channels, batchSize, flatBatchSize, flatSize;
        public readonly int[] n;
        public readonly int[] nF;

        public ShapeInfo(params int[] n)
        {
            this.rank = (byte)n.Length;
            this.n = new int[rank];
            this.nF = new int[rank];
            for (int i = 0; i < n.Length; i++)
            {
                this.n[i] = n[i];
                this.nF[i] = 1;
            }

            nF[^1] = n[^1];

            for (int i = 2; i <= rank; i++)
            {
                nF[^i] = n[^i] * nF[^(i - 1)];
            }

            flatSize = nF[0];

            this.batchSize = 1;
            this.channels = 0;
            this.xLength = 0;
            this.yLength = 0;
            this.zLength = 0;
            this.wLength = 0;
            n0 = 0; n1 = 0; n2 = 0; n3 = 0; n4 = 0; n5 = 0;
            nF0 = 0; nF1 = 0; nF2 = 0; nF3 = 0; nF4 = 0; nF5 = 0;

            switch (rank)
            {
                case 1:
                    n0 = n[0]; nF0 = nF[0];
                    xLength = n[0];
                    break;

                case 2:
                    n0 = n[0]; nF0 = nF[0];
                    n1 = n[1]; nF1 = nF[1];
                    batchSize = n[0];
                    xLength = n[1];
                    break;

                case 3:
                    n0 = n[0]; nF0 = nF[0];
                    n1 = n[1]; nF1 = nF[1];
                    n2 = n[2]; nF2 = nF[2];
                    batchSize = n[0];
                    channels = n[1];
                    xLength = n[2];
                    break;

                case 4:
                    n0 = n[0]; nF0 = nF[0];
                    n1 = n[1]; nF1 = nF[1];
                    n2 = n[2]; nF2 = nF[2];
                    n3 = n[3]; nF3 = nF[3];
                    batchSize = n[0];
                    channels = n[1];
                    yLength = n[2];
                    xLength = n[3];
                    break;

                case 5:
                    n0 = n[0]; nF0 = nF[0];
                    n1 = n[1]; nF1 = nF[1];
                    n2 = n[2]; nF2 = nF[2];
                    n3 = n[3]; nF3 = nF[3];
                    n4 = n[4]; nF4 = nF[4];
                    batchSize = n[0];
                    channels = n[1];
                    zLength = n[2];
                    yLength = n[3];
                    xLength = n[4];
                    break;

                default:
                    batchSize = n[0];
                    channels = n[1];
                    wLength = n[2];
                    zLength = n[3];
                    yLength = n[4];
                    xLength = n[5];
                    n0 = n[0]; nF0 = nF[0];
                    n1 = n[1]; nF1 = nF[1];
                    n2 = n[2]; nF2 = nF[2];
                    n3 = n[3]; nF3 = nF[3];
                    n4 = n[4]; nF4 = nF[4];
                    n5 = n[5]; nF5 = nF[5];
                    break;

            }

            flatBatchSize = flatSize / batchSize;

            if (rank == 1) batchSize = 0;

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

        public ShapeInfo Change(params (int axis, int newSize)[] mods)
        {
            int[] newSh = new int[rank];
            n.CopyTo(newSh, 0);
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
            get => n[ax];
        }

        public static implicit operator Tensor(ShapeInfo a)
        => new Tensor(a).Fill(0);
        public static implicit operator int(ShapeInfo a)
        => a.flatSize;
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
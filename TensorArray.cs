using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace NeuralNetwork
{
	public unsafe abstract class Tensor
	{
		public readonly ShapeInfo shape;
		protected GCHandle pin;
		protected float* ptr;

		public Tensor(int batchSize, int xLength,
		int channels = 0, int yLength = 0, int zLength = 0, int wLength = 0)
		{
			shape = new ShapeInfo(batchSize, xLength, channels, yLength, zLength, wLength);
		}

		~Tensor()
		{
			pin.Free();
		}

		protected void PinAndRef(object array)
		{
			pin = GCHandle.Alloc(array, GCHandleType.Pinned);

			ptr = (float*)pin.AddrOfPinnedObject();
		}

		public static Tensor Create(ShapeInfo shape)
		{
			switch (shape.rank)
			{
				case 1:
					return new Tensor1(shape.batchSize, shape.xLength);

				case 2:
					return new Tensor2(shape.batchSize, shape.channels, shape.xLength);

				case 3:
					return new Tensor3(shape.batchSize, shape.channels, shape.yLength, shape.xLength);

				case 4:
					return new Tensor4(shape.batchSize, shape.channels, shape.zLength, shape.yLength, shape.xLength);

				case 5:
					return new Tensor5(shape.batchSize, shape.channels, shape.wLength, shape.zLength, shape.yLength, shape.xLength);
			}

			return null;
		}

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

		public static Tensor CutBatch(Tensor tensor)
		{
			Tensor newTensor = Create(tensor.shape.Change(batchSize: 1));

			tensor.CopyTo(newTensor);

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

		public void Fill(float @const)
		{
			for (int i = 0; i < this.shape.flatSize; i++)
				this[i] = @const;
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
			int rank = array.Rank;

			Tensor.ShapeInfo shape = new Tensor.ShapeInfo(1, array.GetLength(rank - 1),
			(rank > 1 ? array.GetLength(0) : 0),
			(rank > 2 ? array.GetLength(rank - 2) : 0),
			(rank > 3 ? array.GetLength(rank - 3) : 0));

			Tensor tensor = Tensor.Create(shape);
			tensor.Assimilate(array);
			return tensor;
		}

		public static Tensor[] GetTrainBatches(Array[] trainData, int miniBatchSize)
		{
			Tensor[] tensorBatches = new Tensor[(int)Math.Ceiling((double)trainData.Length / miniBatchSize)];

			int lastBatchSize = trainData.Length % miniBatchSize;

			int rank = trainData[0].Rank;

			Tensor.ShapeInfo shape = new Tensor.ShapeInfo(miniBatchSize, trainData[0].GetLength(rank - 1),
			(rank > 1 ? trainData[0].GetLength(0) : 0),
			(rank > 2 ? trainData[0].GetLength(rank - 2) : 0),
			(rank > 3 ? trainData[0].GetLength(rank - 3) : 0));

			float*[] ptrArray = new float*[trainData.Length];

			for (int t = 0; t < trainData.Length; t++)
			{
				GCHandle pin = GCHandle.Alloc(trainData[t], GCHandleType.Pinned);
				ptrArray[t] = (float*)pin.AddrOfPinnedObject();
			}

			for (int tt = 0; tt < tensorBatches.Length - 1; tt++)
			{
				tensorBatches[tt] = Tensor.Create(shape);
				for (int b = 0, s = 0; b < miniBatchSize; b++)
				{
					for (int sb = 0; sb < shape.flatBatchSize; sb++, s++)
						tensorBatches[tt][s] = ptrArray[tt * miniBatchSize][sb];
				}
			}

			if (lastBatchSize != 0)
			{
				var lastShape = shape.Change(lastBatchSize);
				tensorBatches[^1] = Tensor.Create(lastShape);
				for (int b = 0, s = 0; b < lastBatchSize; b++)
				{
					for (int sb = 0; sb < lastShape.flatBatchSize; sb++, s++)
						tensorBatches[^1][s] = ptrArray[^(lastBatchSize - b)][sb];
				}
			}
			else
			{
				tensorBatches[^1] = Tensor.Create(shape);
				for (int b = 0, s = 0; b < miniBatchSize; b++)
				{
					for (int sb = 0; sb < shape.flatBatchSize; sb++, s++)
						tensorBatches[^1][s] = ptrArray[^(miniBatchSize - b)][sb];
				}
			}

			return tensorBatches;
		}

		public float this[int flatIndex]
		{
			get => ptr[flatIndex];
			set => ptr[flatIndex] = value;
		}

		public float this[int batch, int flat]
		{
			get => ptr[batch * shape.flatBatchSize + flat];
			set => ptr[batch * shape.flatBatchSize + flat] = value;
		}

		public virtual float this[int batch, int channel, int x]
		{
			get => -1;
			set { }
		}

		public virtual float this[int batch, int channel, int y, int x]
		{
			get => -1;
			set { }
		}

		public virtual float this[int batch, int channel, int z, int y, int x]
		{
			get => -1;
			set { }
		}

		public virtual float this[int batch, int channel, int w, int z, int y, int x]
		{
			get => -1;
			set { }
		}

		public readonly struct ShapeInfo
		{
			public readonly int xLength, yLength, zLength, wLength,
			channels, batchSize, flatBatchSize, flatSize;
			public readonly byte rank;

			public ShapeInfo(int batchSize, int xLength, int channels = 0,
			int yLength = 0, int zLength = 0, int wLength = 0)
			{
				this.batchSize = batchSize;
				this.xLength = xLength;
				this.yLength = yLength;
				this.zLength = zLength;
				this.wLength = wLength;
				this.channels = channels;

				byte rank = 1;

				if (channels >= 1) rank++;
				else
					channels = 1;
				if (yLength >= 1) rank++;
				else yLength = 1;
				if (zLength >= 1) rank++;
				else zLength = 1;
				if (wLength >= 1) rank++;
				else wLength = 1;

				this.flatBatchSize = xLength * yLength *
				zLength * wLength * channels;
				this.flatSize = flatBatchSize * batchSize;

				this.rank = rank;
			}

			public ShapeInfo Change(int batchSize = -1, int channels = -1,
			int wLength = -1, int zLength = -1, int yLength = -1, int xLength = -1)
			{
				return new ShapeInfo(
					(batchSize != -1 ? batchSize : this.batchSize),
					(xLength != -1 ? xLength : this.xLength),
					(channels != -1 ? channels : this.channels),
					(yLength != -1 ? yLength : this.yLength),
					(zLength != -1 ? zLength : this.zLength),
					(wLength != -1 ? wLength : this.wLength));
			}

			public static ShapeInfo GetInfo(Array array)
			{
				switch (array.Rank)
				{
					case 1:
						return new ShapeInfo(1, array.Length);

					case 2:
						return new ShapeInfo(array.GetLength(0), array.GetLength(1));

					case 3:
						return new ShapeInfo(array.GetLength(0), array.GetLength(2), array.GetLength(1));

					case 4:
						return new ShapeInfo(array.GetLength(0), array.GetLength(3), array.GetLength(1), array.GetLength(2));

					case 5:
						return new ShapeInfo(array.GetLength(0), array.GetLength(4), array.GetLength(1), array.GetLength(3), array.GetLength(2));

					case 6:
						return new ShapeInfo(array.GetLength(0), array.GetLength(5), array.GetLength(1),
						array.GetLength(4), array.GetLength(3), array.GetLength(2));

					default:
						throw new Exception();
				}
			}

			public static implicit operator int(ShapeInfo info) => info.flatBatchSize;
		}

		public abstract Array ToArray();
	}

	public unsafe sealed class Tensor1 : Tensor
	{
		float[,] array1;

		public Tensor1(int batchSize, int xLength) : base(batchSize, xLength)
		{
			array1 = new float[batchSize, xLength];

			PinAndRef(array1);
		}

		public sealed override Array ToArray() => array1;
	}

	public unsafe sealed class Tensor2 : Tensor
	{
		float[,,] array2;

		public Tensor2(int batchSize, int channels, int xLength) : base(batchSize, xLength, channels)
		{
			array2 = new float[batchSize, channels, xLength];

			PinAndRef(array2);
		}

		public sealed override float this[int batch, int channel, int x]
		{
			get => array2[batch, channel, x];
			set => array2[batch, channel, x] = value;
		}

		public sealed override Array ToArray() => array2;
	}

	public unsafe sealed class Tensor3 : Tensor
	{
		float[,,,] array3;

		public Tensor3(int batchSize, int channels, int yLength, int xLength) : base(batchSize, xLength, channels, yLength)
		{
			array3 = new float[batchSize, channels, yLength, xLength];

			PinAndRef(array3);
		}

		public sealed override float this[int batch, int channel, int y, int x]
		{
			get => array3[batch, channel, y, x];
			set => array3[batch, channel, y, x] = value;
		}

		public sealed override Array ToArray() => array3;
	}

	public unsafe sealed class Tensor4 : Tensor
	{
		float[,,,,] array4;

		public Tensor4(int batchSize, int channels, int zLength, int yLength, int xLength) : base(batchSize, xLength, channels, yLength, zLength)
		{
			array4 = new float[batchSize, channels, zLength, yLength, xLength];

			PinAndRef(array4);
		}

		public sealed override float this[int batch, int channel, int z, int y, int x]
		{
			get => array4[batch, channel, z, y, x];
			set => array4[batch, channel, z, y, x] = value;
		}

		public sealed override Array ToArray() => array4;
	}

	public unsafe sealed class Tensor5 : Tensor
	{
		float[,,,,,] array5;

		public Tensor5(int batchSize, int channels, int wLength, int zLength,
		int yLength, int xLength) : base(batchSize, xLength, channels, yLength, zLength, wLength)
		{
			array5 = new float[batchSize, channels, wLength, zLength, yLength, xLength];

			PinAndRef(array5);
		}

		public sealed override float this[int batch, int channel, int w, int z, int y, int x]
		{
			get => array5[batch, channel, w, z, y, x];
			set => array5[batch, channel, w, z, y, x] = value;
		}

		public sealed override Array ToArray() => array5;
	}
}
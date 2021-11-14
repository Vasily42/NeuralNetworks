using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe class Pooling2D : Layer
	{
		private delegate void PoolingMethod(int batch);

		private PoolingMethod forwardMethod, backPropMethod;

		private (ushort y, ushort x)[,,,] indexedOut;
		private readonly (byte x, byte y) strides;
		private readonly (byte x, byte y) sizeOfPooling;
		private string method;
		private Padding2D paddingLayer;

		public Pooling2D(string method,
		(byte x, byte y) sizeOfPooling,
		(byte x, byte y) strides,
		string padding = "valid")
		{
			this.method = method;
			this.sizeOfPooling = sizeOfPooling;
			this.strides = strides;
			if (padding != "valid")
				paddingLayer = new Padding2D(padding, sizeOfPooling, strides);
		}

		public sealed override void Init()
		{
			if (paddingLayer != null)
				base.InsertBefore(paddingLayer);

			InitDelegates();

			int outXLength = (int)((inputShape.xLength - sizeOfPooling.x) / (float)strides.x + 1);
			int outYLength = (int)((inputShape.yLength - sizeOfPooling.y) / (float)strides.y + 1);

			outputShape = inputShape.Change(xLength: outXLength, yLength: outYLength);

			input = Tensor.Create(inputShape);

			inputDerivatives = Tensor.Create(inputShape);
			inputDerivatives.Fill(0);

			output = Tensor.Create(outputShape);

			outputDerivatives = Tensor.Create(outputShape);

			indexedOut = new (ushort y, ushort x)[inputShape.batchSize, inputShape.channels,
			outYLength, outXLength];
		}

		private void InitDelegates()
		{
			switch (method.ToLower())
			{
				case "max":
					forwardMethod += Max;
					backPropMethod += BackMaxMin;
					break;

				case "min":
					forwardMethod += Min;
					backPropMethod += BackMaxMin;
					break;

				case "average":
					forwardMethod += Average;
					backPropMethod += BackAverage;
					break;
			}
		}

		protected sealed override void ForwardAction(int batch) => forwardMethod(batch);

		private void Max(int batch)
		{
			float max = 0;
			for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
					for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
					{
						for (int y = 0; y < sizeOfPooling.y; y++)
							for (int x = 0; x < sizeOfPooling.x; x++)
							{
								if (x == 0 && y == 0)
								{
									max = input[batch, inputChannel, i, j];
									indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)i, (ushort)j);
									continue;
								}
								if (input[batch, inputChannel, i + y, j + x] > max)
								{
									max = input[batch, inputChannel, i + y, j + x];
									indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)(i + y), (ushort)(j + x));
								}
							}
						output[batch, inputChannel, iOut, jOut] = max;
					}
		}

		private void Min(int batch)
		{
			float min = 0;
			for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
					for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
					{
						for (int y = 0; y < sizeOfPooling.y; y++)
							for (int x = 0; x < sizeOfPooling.x; x++)
							{
								if (x == 0 && y == 0)
								{
									min = input[batch, inputChannel, i, j];
									indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)(i), (ushort)(j));
									continue;
								}
								if (input[batch, inputChannel, i + y, j + x] < min)
								{
									min = input[batch, inputChannel, i + y, j + x];
									indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)(i + y), (ushort)(j + x));
								}
							}
						output[batch, inputChannel, iOut, jOut] = min;
					}
		}

		private void Average(int batch)
		{
			float sum;
			for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
					for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
					{
						sum = 0;
						for (int y = 0; y < sizeOfPooling.y; y++)
							for (int x = 0; x < sizeOfPooling.x; x++)
							{
								sum += input[batch, inputChannel, i + y, j + x];
							}
						output[batch, inputChannel, iOut, jOut] = sum / (sizeOfPooling.y * sizeOfPooling.x);
					}
		}

		protected sealed override void BackPropAction(int batch) => backPropMethod(batch);

		private void BackMaxMin(int batch)
		{
			for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				for (int iOut = 0; iOut < outputShape.yLength; iOut++)
					for (int jOut = 0; jOut < outputShape.xLength; jOut++)
					{
						inputDerivatives[batch, inputChannel, indexedOut[batch, inputChannel, iOut, jOut].y,
						indexedOut[batch, inputChannel, iOut, jOut].x] += outputDerivatives[batch, inputChannel, iOut, jOut];
					}
		}

		private void BackAverage(int batch)
		{
			float averageDelta;

			for (int inputChannel = 0; inputChannel < outputShape.channels; inputChannel++)
				for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
					for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
					{
						averageDelta = outputDerivatives[batch, inputChannel, iOut, jOut] / (sizeOfPooling.x * sizeOfPooling.y);
						for (int y = 0; y < sizeOfPooling.y; y++)
							for (int x = 0; x < sizeOfPooling.x; x++)
							{
								inputDerivatives[batch, inputChannel, i + y, j + x] += averageDelta;
							}
					}
		}
	}
}
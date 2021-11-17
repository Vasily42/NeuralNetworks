using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe class Convolution2D : CalcLayer
	{
		private Parameter[,,,] filters;
		private Parameter[] bias;

		private readonly int numberOfFilters;
		internal readonly (byte x, byte y) strides, kernelSize;
		private Padding2D paddingLayer;

		public Convolution2D(int numberOfFilters,
		(byte x, byte y) kernelSize,
		(byte x, byte y) strides,
		string activationFunction = "linear",
		string padding = "same",
		string parameterInitialization = "kaiming",
		bool NonTrainable = false) : base(activationFunction, parameterInitialization, NonTrainable)
		{
			this.numberOfFilters = numberOfFilters;
			this.kernelSize = kernelSize;
			this.strides = strides;
			if (padding != "valid")
			{
				paddingLayer = new Padding2D(padding, kernelSize, strides);
                Apply(paddingLayer);
			}
		}

		public sealed override void Init()
		{
			int outXLength = (int)((inputShape.xLength - kernelSize.x) / (float)strides.x + 1);
			int outYLength = (int)((inputShape.yLength - kernelSize.y) / (float)strides.y + 1);

			outputShape = inputShape.Change(
			channels: numberOfFilters, xLength: outXLength, yLength: outYLength);

			fanIn = (int)(kernelSize.x * kernelSize.y) * inputShape.channels;
			fanOut = (int)(kernelSize.x * kernelSize.x) * outputShape.channels;

			inputDerivatives = Tensor.Create(inputShape);
			input = Tensor.Create(inputShape);
			output = Tensor.Create(outputShape);
			outputDerivatives = Tensor.Create(outputShape);
			filters = new Parameter[numberOfFilters, inputShape.channels, kernelSize.y, kernelSize.x];
			bias = new Parameter[outputShape.channels];

			for (int filter = 0; filter < numberOfFilters; filter++)
				for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
					for (int y = 0; y < kernelSize.y; y++)
						for (int x = 0; x < kernelSize.x; x++)
						{
							filters[filter, inputChannel, y, x] = new Parameter(randomInitNum());
						}

			for (int filter = 0; filter < outputShape.channels; filter++)
			{
				bias[filter] = new Parameter(0);
			}
		}

		protected sealed override void ForwardAction(int batch)
		{
			float sum = 0;
			for (int filter = 0; filter < outputShape.channels; filter++)
				for (int iOut = 0; iOut < outputShape.yLength; iOut++)
					for (int jOut = 0; jOut < outputShape.xLength; jOut++)
					{
						this.output[batch, filter, iOut, jOut] = bias[filter].value;
					}

			for (int filter = 0; filter < numberOfFilters; filter++)
				for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				{
					for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
						for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
						{
							for (int y = 0; y < kernelSize.y; y++)
								for (int x = 0; x < kernelSize.x; x++)
								{
									sum += filters[filter, inputChannel, y, x].value *
									this.input[batch, inputChannel, i + y, j + x];
								}
							this.output[batch, filter, iOut, jOut] += sum;
							sum = 0;
						}
				}
		}

		protected sealed override void BackPropAction(int batch)
		{
			for (int filter = 0; filter < numberOfFilters; filter++)
				for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				{
					for (int i = 0, iOut = 0; iOut < outputShape.yLength; i += strides.y, iOut++)
						for (int j = 0, jOut = 0; jOut < outputShape.xLength; j += strides.x, jOut++)
						{
							for (int y = 0; y < kernelSize.y; y++)
								for (int x = 0; x < kernelSize.x; x++)
								{
									filters[filter, inputChannel, y, x].gradient += outputDerivatives[batch, filter, iOut, jOut] * this.input[batch, inputChannel, i + y, j + x];
									inputDerivatives[batch, inputChannel, i + y, j + x] += outputDerivatives[batch, filter, iOut, jOut] * filters[filter, inputChannel, y, x].value;
								}
						}
				}

			for (int filter = 0; filter < outputShape.channels; filter++)
				for (int iOut = 0; iOut < outputShape.yLength; iOut++)
					for (int jOut = 0; jOut < outputShape.xLength; jOut++)
					{
						bias[filter].gradient += outputDerivatives[batch, filter, iOut, jOut];
					}
		}

		public sealed override void Correction(Optimizer optimizer, Regularization regularizer)
		{
			if (NonTrainable) return;

			for (int filter = 0; filter < numberOfFilters; filter++)
			{
				for (int inputChannel = 0; inputChannel < inputShape.channels; inputChannel++)
				{
					for (int y = 0; y < kernelSize.y; y++)
						for (int x = 0; x < kernelSize.x; x++)
						{
							regularizer?.GradPenalty(ref filters[filter, inputChannel, y, x]);
							optimizer.Update(ref filters[filter, inputChannel, y, x]);
						}
				}
				regularizer?.GradPenalty(ref bias[filter]);
				optimizer.Update(ref bias[filter]);
			}
		}
	}
}
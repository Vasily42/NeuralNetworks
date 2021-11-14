using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe class Padding2D : Layer
	{
		private readonly float paddingConst;
		private (byte x, byte y) paddingShiftStart, paddingShiftEnd;
		private readonly (byte x, byte y) kernelSize, strides;
		string method;

		public Padding2D(string method, (byte x, byte y) kernelSize,
		(byte x, byte y) strides, float paddingConst = 0)
		{
			this.paddingConst = paddingConst;
			this.method = method;
			this.kernelSize = kernelSize;
			this.strides = strides;
		}

		public sealed override void Init()
		{
			switch (method.ToLower())
			{
				case "same":
					paddingShiftStart.x = (byte)Math.Floor((strides.x * Math.Ceiling(inputShape.xLength / (float)strides.x) - inputShape.xLength + kernelSize.x - strides.x) / 2f);
					paddingShiftStart.y = (byte)Math.Floor((strides.y * Math.Ceiling(inputShape.yLength / (float)strides.y) - inputShape.yLength + kernelSize.y - strides.y) / 2f);
					paddingShiftEnd.x = (byte)Math.Ceiling((strides.x * Math.Ceiling(inputShape.xLength / (float)strides.x) - inputShape.xLength + kernelSize.x - strides.x) / 2f);
					paddingShiftEnd.y = (byte)Math.Ceiling((strides.y * Math.Ceiling(inputShape.yLength / (float)strides.y) - inputShape.yLength + kernelSize.y - strides.y) / 2f);
					break;

				case "full":
					paddingShiftStart.x = (byte)(kernelSize.x - 1);
					paddingShiftStart.y = (byte)(kernelSize.y - 1);
					paddingShiftEnd.x = (byte)(kernelSize.x - 1);
					paddingShiftEnd.y = (byte)(kernelSize.y - 1);
					break;
			}

			outputShape = inputShape.Change(xLength:
			inputShape.xLength + paddingShiftStart.x + paddingShiftEnd.x,
			yLength: inputShape.yLength + paddingShiftStart.y + paddingShiftEnd.y);

			input = Tensor.Create(inputShape);
			inputDerivatives = Tensor.Create(inputShape);
			output = Tensor.Create(outputShape);
			outputDerivatives = Tensor.Create(outputShape);
			output.Fill(paddingConst);
		}

		protected sealed override void ForwardAction(int batch)
		{
			for (int channel = 0, s = 0; channel < inputShape.channels; channel++)
				for (int i = 0; i < inputShape.yLength; i++)
					for (int j = 0; j < inputShape.xLength; j++, s++)
						this.output[batch, channel, i + paddingShiftStart.y, j + paddingShiftStart.x] =
						this.input[batch, s];
		}

		protected sealed override void BackPropAction(int batch)
		{
			for (int channel = 0, s = 0; channel < inputShape.channels; channel++)
				for (int i = 0; i < inputShape.yLength; i++)
					for (int j = 0; j < inputShape.xLength; j++, s++)
						this.inputDerivatives[batch, channel, i, j] = outputDerivatives[batch, channel, i + paddingShiftStart.y, j + paddingShiftStart.x];
		}
	}
}
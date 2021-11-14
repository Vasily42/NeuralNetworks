using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe class Flatten : Layer
	{
		public sealed override void Init()
		{
			outputShape = new Tensor.ShapeInfo(inputShape.batchSize, inputShape.flatBatchSize);
			output = Tensor.Create(outputShape);
			outputDerivatives = Tensor.Create(outputShape);
			input = Tensor.Create(inputShape);
			inputDerivatives = Tensor.Create(inputShape);
		}

		protected sealed override void ForwardAction(int batch)
		{
			for (int i = 0; i < outputShape.flatBatchSize; i++)
				output[batch, i] = input[batch, i];
		}

		protected sealed override void BackPropAction(int batch)
		{
			for (int i = 0; i < inputShape.flatBatchSize; i++)
				inputDerivatives[batch, i] = outputDerivatives[batch, i];
		}
	}
}
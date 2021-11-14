using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe class Input : Layer
	{
		public Input(Tensor.ShapeInfo shape)
		{
			this.inputShape = shape;
			outputShape = inputShape;
		}

		public void Forward(Tensor input, bool training)
		{
			int actualMBSize = input.shape.batchSize;
			if (input.shape.rank == inputShape.rank)
				nextLayer.Forward(input, in actualMBSize, in training);
			else nextLayer.Forward(Tensor.Reshape(inputShape, input), in actualMBSize, in training);
		}

		public void Forward(dynamic input, bool training)
		{
			Tensor tensor = Tensor.Create(inputShape);
			tensor.Assimilate(input);
			Forward(tensor, training);
		}

		public sealed override void BackProp(Tensor deriv, in int actualMBSize) { }
	}
}
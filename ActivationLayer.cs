using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe abstract class ActivationLayer : Layer
	{
		public override void Init()
		{
			outputShape = inputShape;

			input = Tensor.Create(inputShape);
			inputDerivatives = Tensor.Create(inputShape);
			output = Tensor.Create(outputShape);
			outputDerivatives = Tensor.Create(outputShape);
		}

		public static ActivationLayer CreateActivation(string name)
		{
			ActivationLayer activation = null;
			switch (name.ToLower())
			{
				case "linear":
					activation = null;
					break;

				case "sigmoid":
					activation = new Sigmoid();
					break;

				case "tangens":
					activation = new Tangens();
					break;

				case "relu":
					activation = new ReLU();
					break;

				case "elu":
					activation = new ELU();
					break;

				case "swish":
					activation = new Swish();
					break;

				case "softmax":
					activation = new Softmax();
					break;
			}
			return activation;
		}

		protected override void ForwardAction(int batch)
		{
			for (int i = 0; i < inputShape.flatBatchSize; i++)
			{
				this.output[batch, i] = Activation(this.input[batch, i]);
			}
		}

		protected override void BackPropAction(int batch)
		{
			for (int i = 0; i < outputShape.flatBatchSize; i++)
				inputDerivatives[batch, i] = outputDerivatives[batch, i]
				* Derivative(this.output[batch, i], this.input[batch, i]);
		}
		
		protected virtual float Activation(float x) => 0;
		protected virtual float Derivative(float y, float x) => 0;
	}

	public class Sigmoid : ActivationLayer
	{
		protected sealed override float Activation(float x)
		{
			return (float)(1f / (1 + Math.Exp(-x)));
		}

		protected sealed override float Derivative(float y, float x)
		{
			return (float)(((1 - y) * y) + 0.02f);
		}
	}

	public class Tangens : ActivationLayer
	{
		protected sealed override float Activation(float x)
		{
			return (float)(2f / (1 + Math.Exp(-2 * x)) - 1);
		}

		protected sealed override float Derivative(float y, float x)
		{
			return ((float)(1 - y * y) + 0.02f);
		}
	}

	public class ReLU : ActivationLayer
	{
		protected sealed override float Activation(float x)
		{
			return Math.Max(0, x);
		}

		protected sealed override float Derivative(float y, float x)
		{
			return (y > 0 ? 1 : 0);
		}
	}

	public class ELU : ActivationLayer
	{
		protected sealed override float Activation(float x)
		{
			return (x >= 0 ? x : 0.103f * (float)(Math.Exp(x) - 1));
		}

		protected sealed override float Derivative(float y, float x)
		{
			return (y >= 0 ? 1 : y + 0.103f);
		}
	}

	public class Swish : ActivationLayer
	{
		protected sealed override float Activation(float x)
		{
			return (float)(x * (1f / (1 + Math.Exp(-x))));
		}

		protected sealed override float Derivative(float y, float x)
		{
			return (float)(y + (1f / (1 + Math.Exp(-x))) * (1 - y));
		}
	}

	public unsafe class Softmax : ActivationLayer
	{
		protected sealed override void ForwardAction(int batch)
		{
			float sum = epsilon;
			for (int i = 0; i < inputShape.flatBatchSize; i++)
			{
				sum += (float)Math.Exp(input[batch, i]);
			}

			for (int i = 0; i < inputShape.flatBatchSize; i++)
			{
				output[batch, i] = (float)(Math.Exp(input[batch, i] + epsilon) / sum);
			}
		}

		protected sealed override void BackPropAction(int batch)
		{
			float sum;

			for (int j = 0; j < inputShape.flatBatchSize; j++)
			{
				sum = 0;
				for (int i = 0; i < inputShape.flatBatchSize; i++)
				{
					if (i == j)
					{
						sum += (output[batch, i] * (1 - output[batch, j])) * outputDerivatives[batch, i];
					}
					else
					{
						sum += (-output[batch, i] * output[batch, j]) * outputDerivatives[batch, i];
					}
				}
				outputDerivatives[batch, j] = sum;
			}
		}
	}
}
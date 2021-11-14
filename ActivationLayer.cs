using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe abstract class ActivationLayer : Layer
	{
		protected delegate float Activation(float x);

		protected delegate float Derivative(float y, float x);

		protected Activation activation;
		protected Derivative derivative;

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
				this.output[batch, i] = activation(this.input[batch, i]);
			}
		}

		protected override void BackPropAction(int batch)
		{
			for (int i = 0; i < outputShape.flatBatchSize; i++)
				inputDerivatives[batch, i] = outputDerivatives[batch, i]
				* derivative(this.output[batch, i], this.input[batch, i]);
		}

		public static float Sigmoid(float x)
		{
			return (float)(1f / (1 + Math.Exp(-x)));
		}

		public static float DSigmoid(float y, float x)
		{
			return (float)(((1 - y) * y) + 0.02f);
		}

		public static float ReLU(float x)
		{
			return Math.Max(0, x);
		}

		public static float DReLU(float y, float x)
		{
			return (y > 0 ? 1 : 0);
		}

		public static float Tangens(float x)
		{
			return (float)(2f / (1 + Math.Exp(-2 * x)) - 1);
		}

		public static float DTangens(float y, float x)
		{
			return ((float)(1 - y * y) + 0.02f);
		}

		public static float ELU(float x)
		{
			return (x >= 0 ? x : 0.103f * (float)(Math.Exp(x) - 1));
		}

		public static float DELU(float y, float x)
		{
			return (y >= 0 ? 1 : y + 0.103f);
		}

		public static float Swish(float x)
		{
			return (float)(x * Sigmoid(x));
		}

		public static float DSwish(float y, float x)
		{
			return y + Sigmoid(x) * (1 - y);
		}
	}

	public class Sigmoid : ActivationLayer
	{
		public Sigmoid()
		{
			activation = ActivationLayer.Sigmoid;
			derivative = ActivationLayer.DSigmoid;
		}
	}

	public class Tangens : ActivationLayer
	{
		public Tangens()
		{
			activation += ActivationLayer.Tangens;
			derivative += ActivationLayer.DTangens;
		}
	}

	public class ReLU : ActivationLayer
	{
		public ReLU()
		{
			activation += ActivationLayer.ReLU;
			derivative += ActivationLayer.DReLU;
		}
	}

	public class ELU : ActivationLayer
	{
		public ELU()
		{
			activation += ActivationLayer.ELU;
			derivative += ActivationLayer.DELU;
		}
	}

	public class Swish : ActivationLayer
	{
		public Swish()
		{
			activation += ActivationLayer.Swish;
			derivative += ActivationLayer.DSwish;
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
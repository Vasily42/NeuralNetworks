using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public unsafe abstract class Loss : Layer
	{
		protected float lossValue;
		public float LossValue => lossValue;

		protected delegate float LossVal(float sum);
		protected delegate float Partial(float ideal, float pred);

		protected LossVal lossGen;
		protected Partial partialDerivative, partialError;

		public Tensor Predicted => input;

		public sealed override void Init()
		{
			input = Tensor.Create(inputShape);
			inputDerivatives = Tensor.Create(inputShape);
		}

		public static Loss CreateLoss(string lossName)
		{
			switch (lossName.ToLower())
			{
				case "mae":
					return new MeanAbsoluteError();

				case "mse":
					return new MeanSquaredError();

				case "cross entropy":
				case "ce":
					return new CrossEntropy();

				default:
					throw new Exception();
			}
		}

		public sealed override void Forward(Tensor input, in int a, in bool t)
		{
			input.CopyTo(this.input);
		}

		public void BackProp(Tensor ideal)
		{
			lossValue = 0;

			int actualMBSize = ideal.shape.batchSize;

			Action<int> derivationAction = (batch) =>
			{
				for (int i = 0; i < inputShape.flatBatchSize; i++)
				{
					this.inputDerivatives[batch, i] = partialDerivative(ideal[batch, i], this.input[batch, i]);
					lossValue += partialError(ideal[batch, i], this.input[batch, i]);
				}
			};

			for (int i = 0; i < actualMBSize; i++)
				derivationAction(i);

			lossValue = lossGen(lossValue);
			lossValue /= actualMBSize;

			prevLayer.BackProp(this.inputDerivatives, in actualMBSize);
		}

		public void BackProp(Array ideal)
		{
			Tensor tensor = Tensor.Create(ideal);
			BackProp(tensor);
		}

		public virtual float ComputeError(Tensor ideal)
		{
			lossValue = 0;

			Action<int> errCalculatingAction = (batch) =>
			{
				for (int i = 0; i < inputShape.flatBatchSize; i++)
					lossValue += partialError(ideal[batch, i], this.input[batch, i]);
			};

			for (int i = 0; i < ideal.shape.batchSize; i++)
				errCalculatingAction(i);

			lossValue = lossGen(lossValue);
			return lossValue;
		}

		public float ComputeError(Array ideal)
		{
			Tensor tensor = Tensor.AddBatchDimension(ideal);
			return ComputeError(tensor);
		}
	}

	public unsafe class MeanSquaredError : Loss
	{
		public MeanSquaredError()
		{
			partialError = Square;
			partialDerivative = SquareDerivative;
			lossGen = Average;
		}

		private float Square(float ideal, float pred) => (float)Math.Pow(ideal - pred, 2);
		private float SquareDerivative(float ideal, float pred) => pred - ideal;
		private float Average(float sum) => sum / inputShape.flatBatchSize;
	}

	public unsafe class MeanAbsoluteError : Loss
	{
		public MeanAbsoluteError()
		{
			partialError = Abs;
			partialDerivative = AbsDerivative;
			lossGen = Average;
		}

		private float Abs(float ideal, float pred) => Math.Abs(ideal - pred);
		private float AbsDerivative(float ideal, float pred) => (ideal - pred > 0 ? -1 : 1);
		private float Average(float sum) => sum / inputShape.flatBatchSize;
	}

	public unsafe class CrossEntropy : Loss
	{
		public CrossEntropy()
		{
			partialError = CEPartial;
			partialDerivative = CEDerivative;
			lossGen = Neg;
		}

		private float CEPartial(float ideal, float pred) => ideal * (float)Math.Log(pred + epsilon);
		private float CEDerivative(float ideal, float pred) => -ideal / (pred + epsilon);
		private float Neg(float sum) => -sum;
	}
}
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
					this.inputDerivatives[batch, i] = PartialDerivative(ideal[batch, i], this.input[batch, i]);
					lossValue += PartialError(ideal[batch, i], this.input[batch, i]);
				}
			};

			for (int i = 0; i < actualMBSize; i++)
				derivationAction(i);

			lossValue = LossGen(lossValue);
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
					lossValue += PartialError(ideal[batch, i], this.input[batch, i]);
			};

			for (int i = 0; i < ideal.shape.batchSize; i++)
				errCalculatingAction(i);

			lossValue = LossGen(lossValue);
			return lossValue;
		}

		public float ComputeError(Array ideal)
		{
			Tensor tensor = Tensor.AddBatchDimension(ideal);
			return ComputeError(tensor);
		}
		
		protected virtual float PartialError(float ideal, float pred) => 0;
		protected virtual float PartialDerivative(float ideal, float pred) => 0;
		protected virtual float LossGen(float loss) => 0;
	}

	public unsafe class MeanSquaredError : Loss
	{
		protected sealed override float PartialError(float ideal, float pred) => (float)Math.Pow(ideal - pred, 2);
		protected sealed override float PartialDerivative(float ideal, float pred) => pred - ideal;
		protected sealed override float LossGen(float sum) => sum / inputShape.flatBatchSize;
	}

	public unsafe class MeanAbsoluteError : Loss
	{
		protected sealed override float PartialError(float ideal, float pred) => Math.Abs(ideal - pred);
		protected sealed override float PartialDerivative(float ideal, float pred) => (float)-Math.Sign(ideal - pred);
		protected sealed override float LossGen(float sum) => sum / inputShape.flatBatchSize;
	}

	public unsafe class CrossEntropy : Loss
	{
		protected sealed override float PartialError(float ideal, float pred) => ideal * (float)Math.Log(pred + epsilon);
		protected sealed override float PartialDerivative(float ideal, float pred) => -ideal / (pred + epsilon);
		protected sealed override float LossGen(float sum) => -sum;
	}
}
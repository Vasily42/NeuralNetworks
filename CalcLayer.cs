using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public unsafe abstract class CalcLayer : Layer
	{
		protected delegate float RandomWeightInit();

		protected RandomWeightInit randomInitNum;

		protected int fanIn, fanOut;

		protected readonly bool NonTrainable;

		protected CalcLayer(string activationFunction, string parameterInitialization, bool NonTrainable)
		{
			InsertActivation(activationFunction);
			this.NonTrainable = NonTrainable;
			switch (parameterInitialization.ToLower())
			{
				case "xavier":
					randomInitNum = Xavier;
					break;

				case "kaiming":
					randomInitNum = Kaiming;
					break;
			}
		}

		public abstract void Correction(Optimizer optimizer, Regularization regularizer);

		private float Xavier()
		{
			return (float)(Math.Sqrt(6f / (fanIn + fanOut)) * (2 * StGeneral.NextFloat() - 1));
		}

		private float Kaiming()
		{
			return (float)(Math.Sqrt(2f / fanIn) * (2 * StGeneral.NextFloat() - 1));
		}
	}
}
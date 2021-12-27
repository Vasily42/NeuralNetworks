global using NeuralNetwork;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public abstract class Model
	{
		public bool Initialized { get; protected set; }

		internal Optimizer optimizer;
		public float LearningRate
		{
			get => optimizer.learningRate;
			set => optimizer.learningRate = value;
		}

		internal Regularization regularizer;

		internal long iteration;
		public long Iteration => iteration;

		public bool Training { get; set; }

		internal readonly Random helpRnd;

		public Model()
		{
			helpRnd = new Random();
		}
	}

	public sealed class Sequential : Model
	{
		public delegate void Render(float value);
		public event Render ErrorRendering, IterationRendering, EpochRendering;

		public Tensor Answer => loss.Predicted;
		public float ComputeError(Array ideal) => loss.ComputeError(ideal);

		private Input firstLayer;
		private Layer tempLayer;
		private Loss loss;
		private int helpIndex = 0;

		public void Add(Layer layer)
		{
			if (helpIndex == 0)
			{
				if (!(layer is Input)) throw new Exception();
				else
				{
					firstLayer = (Input)layer;
				}
				tempLayer = layer.Leave;
				helpIndex++;
				return;
			}

			tempLayer = layer.Apply(tempLayer);
		}

		public void Init(dynamic optimizer, dynamic loss = null, Regularization regularizer = null)
		{
			if (optimizer is string)
				this.optimizer = Optimizer.GetOptimizer(optimizer);
			else
				this.optimizer = optimizer;

			this.optimizer.@base = this;

			if (!(tempLayer is Loss))
			{
				if (loss is string)
					this.loss = Loss.CreateLoss(loss);
				else
					this.loss = loss;

				this.loss.Apply(tempLayer);
			}
			else loss = tempLayer;

			this.regularizer = regularizer;

			firstLayer.InitGraph();

			Training = true;
			Initialized = true;
		}

		public void ForwardBatch(Tensor input)
		{
			firstLayer.Forward(input, Training);
		}

		public Tensor Forward(Array input)
		{
			firstLayer.Forward(Tensor.
			AddBatchDimension(input), Training);
			return loss.Predicted;
		}

		public void BackPropBatch(Tensor ideal)
		{
			loss.BackProp(ideal);

			ErrorRendering?.Invoke(loss.LossValue);

			firstLayer.ParameterCorrection(optimizer, regularizer);

			iteration++;
		}

		public void BackProp(Array ideal)
		{
			BackPropBatch(Tensor.AddBatchDimension(ideal));
		}

		public void Train(Tensor[] forwardData, Tensor[] backPropData, int epochsToTrain = 1)
		{
			Training = true;

			for (int epoch = 0; epoch < epochsToTrain; epoch++)
			{
				for (int i = 0; i < forwardData.Length; i++)
				{
					this.ForwardBatch(forwardData[i]);
					this.BackPropBatch(backPropData[i]);
					IterationRendering?.Invoke(epoch * forwardData.Length + i);
				}
				EpochRendering?.Invoke(epoch);
			}

			Training = false;
		}

		public void Train(Array[] forwardData, Array[] backPropData, int epochsToTrain = 1)
		{
			Tensor[] batchesForward = Tensor.GetTrainBatches(forwardData, firstLayer.inputShape.batchSize);
			Tensor[] batchesBackProp = Tensor.GetTrainBatches(backPropData, firstLayer.inputShape.batchSize);

			Train(batchesForward, batchesBackProp, epochsToTrain);
		}

		public (double accuracy, double lossValue) OneHotTest(Tensor[] forwardData, Tensor[] idealBackPropData)
		{
			int pass = 0;

			double lossValueAverage = 0, accuracy;

			for (int i = 0; i < forwardData.Length; i++)
			{
				firstLayer.Forward(forwardData[i], false);

				if (loss.Predicted.IndexOfMax() == idealBackPropData[i].IndexOfMax())
					pass++;

				lossValueAverage += loss.ComputeError(idealBackPropData[i]);
			}

			lossValueAverage /= forwardData.Length;
			accuracy = (double)pass / forwardData.Length;

			return (accuracy, lossValueAverage);
		}
	}
}

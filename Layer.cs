using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public abstract class Layer
	{
		protected const float epsilon = 1.0E-8F;

		internal Layer nextLayer, prevLayer;

		protected Tensor input, output, inputDerivatives, outputDerivatives;

		internal Tensor.ShapeInfo inputShape, outputShape;

		public Layer Apply(Layer prevLayer)
		{
			this.prevLayer = prevLayer;
			prevLayer.nextLayer = this;
			return this;
		}

		public virtual void Init() { }

		public void InitGraph()
		{
			if (prevLayer != null)
				this.inputShape = prevLayer.outputShape;

			Init();
			nextLayer?.InitGraph();
		}

		public void ParameterCorrection(Optimizer optimizer, Regularization regularizer)
		{
			if (this is CalcLayer)
			{
				CalcLayer calc = (CalcLayer)this;
				calc.Correction(optimizer, regularizer);
			}
			nextLayer?.ParameterCorrection(optimizer, regularizer);
		}

		protected void InsertAhead(Layer layer)
		{
			if (nextLayer != null)
			{
				nextLayer.prevLayer = layer;
				layer.nextLayer = this.nextLayer;
			}
			layer.prevLayer = this;
			nextLayer = layer;
		}

		protected void InsertBefore(Layer layer)
		{
			if (this is Input) return;
			layer.inputShape = prevLayer.outputShape;
			prevLayer.nextLayer = layer;
			layer.nextLayer = this;
			layer.prevLayer = prevLayer;
			prevLayer = layer;
			prevLayer.Init();
			this.inputShape = prevLayer.outputShape;
		}

		protected void InsertActivation(string activationName = null)
		{
			if (activationName != null && activationName.ToLower() != "linear")
				InsertAhead(ActivationLayer.CreateActivation(activationName));
		}

		public virtual void Forward(Tensor input, in int actualMBSize, in bool training)
		{
			input.CopyTo(this.input, actualMBSize);
			for (int i = 0; i < actualMBSize; i++)
				ForwardAction(i);
			nextLayer?.Forward(this.output, in actualMBSize, in training);
		}

		protected virtual void ForwardAction(int batch) { }

		public virtual void BackProp(Tensor deriv, in int actualMBSize)
		{
			deriv.CopyTo(outputDerivatives, actualMBSize);
			for (int i = 0; i < actualMBSize; i++)
				BackPropAction(i);
			prevLayer?.BackProp(inputDerivatives, in actualMBSize);
		}

		protected virtual void BackPropAction(int batch) { }
	}
}
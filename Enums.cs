namespace NeuralNetwork;

public enum ActivationFunction
{
    Sigmoid, 
    Tanh,
    ReLU,
    ELU,
    Swish,
    Softmax
}

public enum OptimizerAlgorithm
{
    SGD,
    Momentum,
    Nesterov,
    Adam,
    Nadam
}

public enum LossFunction
{
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    CategoricalCrossEntropy
}
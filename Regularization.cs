using System.Text.RegularExpressions;

namespace NeuralNetwork;

public class Regularization
{
    private readonly float l1, l2;

    public Regularization(float l1, float l2)
    {
        this.l1 = l1;
        this.l2 = l2;
    }

    public void GradPenalty(Tensor weights, Tensor gradient)
    {
        for (int i = 0; i < weights.shape.nF0; i++)
        {
            gradient[i] += l2 * weights[i];
            gradient[i] += Math.Sign(weights[i]) * l1;
        }
    }

    public static implicit operator Regularization((float l1, float l2) tupl) => new Regularization(tupl.l1, tupl.l2);
}

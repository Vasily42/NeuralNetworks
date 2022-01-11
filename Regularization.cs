namespace NeuralNetwork;

public class Regularization
{
    private readonly float l1, l2;

    public Regularization(float l1, float l2)
    {
        this.l1 = l1;
        this.l2 = l2;
    }

    public void GradPenalty(ref Parameter p)
    {
        p.gradient += l2 * p.value;
        p.gradient += Math.Sign(p.value) * l1;
    }
}

namespace NeuralNetwork;

[Serializable]
public unsafe class Padding2D : Layer
{
    private readonly float paddingConst;
    private (byte x, byte y) paddingShiftStart, paddingShiftEnd;
    private readonly (byte x, byte y) kernelSize, strides;
    readonly string method;

    public Padding2D(string method, (byte x, byte y) kernelSize,
    (byte x, byte y) strides, float paddingConst = 0, string name = null) : base(name)
    {
        this.paddingConst = paddingConst;
        this.method = method;
        this.kernelSize = kernelSize;
        this.strides = strides;
    }

    public sealed override void Init(Optimizer optimizer)
    {
        switch (method.ToLower())
        {
            case "same":
                paddingShiftStart.x = (byte)Math.Floor((strides.x * Math.Ceiling(inputShape.n3 / (float)strides.x) - inputShape.n3 + kernelSize.x - strides.x) / 2f);
                paddingShiftStart.y = (byte)Math.Floor((strides.y * Math.Ceiling(inputShape.n2 / (float)strides.y) - inputShape.n2 + kernelSize.y - strides.y) / 2f);
                paddingShiftEnd.x = (byte)Math.Ceiling((strides.x * Math.Ceiling(inputShape.n3 / (float)strides.x) - inputShape.n3 + kernelSize.x - strides.x) / 2f);
                paddingShiftEnd.y = (byte)Math.Ceiling((strides.y * Math.Ceiling(inputShape.n2 / (float)strides.y) - inputShape.n2 + kernelSize.y - strides.y) / 2f);
                break;

            case "full":
                paddingShiftStart.x = (byte)(kernelSize.x - 1);
                paddingShiftStart.y = (byte)(kernelSize.y - 1);
                paddingShiftEnd.x = (byte)(kernelSize.x - 1);
                paddingShiftEnd.y = (byte)(kernelSize.y - 1);
                break;
        }

        outputShape = inputShape.Change((3,
        inputShape.n3 + paddingShiftStart.x + paddingShiftEnd.x),
        (2, inputShape.n2 + paddingShiftStart.y + paddingShiftEnd.y));

        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);
        output.Fill(paddingConst);
    }

    protected sealed override void ForwardAction(int batch)
    {
        for (int channel = 0, s = 0; channel < inputShape.n1; channel++)
            for (int i = 0; i < inputShape.n2; i++)
                for (int j = 0; j < inputShape.n3; j++, s++)
                    this.output[batch, channel, i + paddingShiftStart.y, j + paddingShiftStart.x] =
                    this.input[batch, s];
    }

    protected sealed override void BackPropAction(int batch)
    {
        for (int channel = 0, s = 0; channel < inputShape.n1; channel++)
            for (int i = 0; i < inputShape.n2; i++)
                for (int j = 0; j < inputShape.n3; j++, s++)
                    this.inputDerivatives[batch, channel, i, j] = outputDerivatives[batch, channel, i + paddingShiftStart.y, j + paddingShiftStart.x];
    }
}

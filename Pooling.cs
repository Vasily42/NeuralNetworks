namespace NeuralNetwork;

[Serializable]
public unsafe class Pooling2D : Layer
{
    private delegate void PoolingMethod(int batch);

    private PoolingMethod forwardMethod, backPropMethod;

    private (ushort y, ushort x)[,,,] indexedOut;
    private readonly (byte x, byte y) strides;
    private readonly (byte x, byte y) sizeOfPooling;
    private readonly string method;
    private readonly Padding2D paddingLayer;

    public Pooling2D(string method,
    (byte x, byte y) sizeOfPooling,
    (byte x, byte y) strides,
    string padding = "valid", string name = null) : base(name)
    {
        this.method = method;
        this.sizeOfPooling = sizeOfPooling;
        this.strides = strides;
        if (padding != "valid")
        {
            paddingLayer = new Padding2D(padding, sizeOfPooling, strides);
            Apply(paddingLayer);
        }
    }

    public sealed override void Init(Optimizer optimizer)
    {
        InitDelegates();

        int outXLength = (int)((inputShape.n3 - sizeOfPooling.x) / (float)strides.x + 1);
        int outYLength = (int)((inputShape.n2 - sizeOfPooling.y) / (float)strides.y + 1);

        outputShape = inputShape.Change((3, outXLength), (2, outYLength));

        input = new Tensor(inputShape);
        inputDerivatives = new Tensor(inputShape);
        output = new Tensor(outputShape);
        outputDerivatives = new Tensor(outputShape);

        inputDerivatives.Fill(0);

        indexedOut = new (ushort y, ushort x)[inputShape.n0, inputShape.n1,
        outYLength, outXLength];
    }

    private void InitDelegates()
    {
        switch (method.ToLower())
        {
            case "max":
                forwardMethod += Max;
                backPropMethod += BackMaxMin;
                break;

            case "min":
                forwardMethod += Min;
                backPropMethod += BackMaxMin;
                break;

            case "average":
                forwardMethod += Average;
                backPropMethod += BackAverage;
                break;
        }
    }

    protected sealed override void ForwardAction(int batch) => forwardMethod(batch);

    private void Max(int batch)
    {
        float max = 0;
        for (int inputChannel = 0; inputChannel < inputShape.n1; inputChannel++)
            for (int i = 0, iOut = 0; iOut < outputShape.n2; i += strides.y, iOut++)
                for (int j = 0, jOut = 0; jOut < outputShape.n3; j += strides.x, jOut++)
                {
                    for (int y = 0; y < sizeOfPooling.y; y++)
                        for (int x = 0; x < sizeOfPooling.x; x++)
                        {
                            if (x == 0 && y == 0)
                            {
                                max = input[batch, inputChannel, i, j];
                                indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)i, (ushort)j);
                                continue;
                            }
                            if (input[batch, inputChannel, i + y, j + x] > max)
                            {
                                max = input[batch, inputChannel, i + y, j + x];
                                indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)(i + y), (ushort)(j + x));
                            }
                        }
                    output[batch, inputChannel, iOut, jOut] = max;
                }
    }

    private void Min(int batch)
    {
        float min = 0;
        for (int inputChannel = 0; inputChannel < inputShape.n1; inputChannel++)
            for (int i = 0, iOut = 0; iOut < outputShape.n2; i += strides.y, iOut++)
                for (int j = 0, jOut = 0; jOut < outputShape.n3; j += strides.x, jOut++)
                {
                    for (int y = 0; y < sizeOfPooling.y; y++)
                        for (int x = 0; x < sizeOfPooling.x; x++)
                        {
                            if (x == 0 && y == 0)
                            {
                                min = input[batch, inputChannel, i, j];
                                indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)(i), (ushort)(j));
                                continue;
                            }
                            if (input[batch, inputChannel, i + y, j + x] < min)
                            {
                                min = input[batch, inputChannel, i + y, j + x];
                                indexedOut[batch, inputChannel, iOut, jOut] = ((ushort)(i + y), (ushort)(j + x));
                            }
                        }
                    output[batch, inputChannel, iOut, jOut] = min;
                }
    }

    private void Average(int batch)
    {
        float sum;
        for (int inputChannel = 0; inputChannel < inputShape.n1; inputChannel++)
            for (int i = 0, iOut = 0; iOut < outputShape.n2; i += strides.y, iOut++)
                for (int j = 0, jOut = 0; jOut < outputShape.n3; j += strides.x, jOut++)
                {
                    sum = 0;
                    for (int y = 0; y < sizeOfPooling.y; y++)
                        for (int x = 0; x < sizeOfPooling.x; x++)
                        {
                            sum += input[batch, inputChannel, i + y, j + x];
                        }
                    output[batch, inputChannel, iOut, jOut] = sum / (sizeOfPooling.y * sizeOfPooling.x);
                }
    }

    protected sealed override void BackPropAction(int batch) => backPropMethod(batch);

    private void BackMaxMin(int batch)
    {
        for (int inputChannel = 0; inputChannel < inputShape.n1; inputChannel++)
            for (int iOut = 0; iOut < outputShape.n2; iOut++)
                for (int jOut = 0; jOut < outputShape.n3; jOut++)
                {
                    inputDerivatives[batch, inputChannel, indexedOut[batch, inputChannel, iOut, jOut].y,
                    indexedOut[batch, inputChannel, iOut, jOut].x] += outputDerivatives[batch, inputChannel, iOut, jOut];
                }
    }

    private void BackAverage(int batch)
    {
        float averageDelta;

        for (int inputChannel = 0; inputChannel < outputShape.n1; inputChannel++)
            for (int i = 0, iOut = 0; iOut < outputShape.n2; i += strides.y, iOut++)
                for (int j = 0, jOut = 0; jOut < outputShape.n3; j += strides.x, jOut++)
                {
                    averageDelta = outputDerivatives[batch, inputChannel, iOut, jOut] / (sizeOfPooling.x * sizeOfPooling.y);
                    for (int y = 0; y < sizeOfPooling.y; y++)
                        for (int x = 0; x < sizeOfPooling.x; x++)
                        {
                            inputDerivatives[batch, inputChannel, i + y, j + x] += averageDelta;
                        }
                }
    }
}

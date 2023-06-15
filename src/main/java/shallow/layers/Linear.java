package shallow.layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.configs.LinearLayerConfig;

public class Linear extends WeightedLayer implements ShapeChangingLayer {
    int units = 20;

    public Linear(LinearLayerConfig config) {
        super(config);
        units = config.getUnits();
    }

    // Instantiates Weight of shape [inShape[0], units] and Bias of shape [1, units]
    @Override
    public void init(long... inShape) {
        if (inShape.length != 1 && inShape.length != 2) {
            throw new IllegalArgumentException();
        }
        int offset = (inShape.length == 1) ? 0 : 1;
        int inputSize = (int) inShape[offset];
        weight.values = Nd4j.create(DataType.FLOAT, inputSize, units);
        weightInitializer.init(inputSize, units, weight.values);
        bias.values = Nd4j.create(DataType.FLOAT, 1, units);
        biasInitializer.init(inputSize, units, bias.values);
        weight.grads = Nd4j.zerosLike(weight.values);
        bias.grads = Nd4j.zerosLike(bias.values);
    }

    @Override
    public long[] getOutputShape() {
        return new long[]{units};
    }

    // computes linear function Z = dot(W,X) + b,
    // where X.shape() = [batch_size, input_size], Z.shape() = [batch_size, output_size]
    @Override
    public INDArray forward(INDArray input) {
        // TODO check shape of input
        currentBatchSize = input.shape()[0];
        input = input.castTo(DataType.FLOAT);
        cache.put("X", input);
        return input.mmul(weight.values).addi(bias.values);
    }

    // computes partial derivatives chained with derivatives[0], derivatives[0].shape() = [batch_size, output_size]
    @Override
    public INDArray backward(INDArray dZ) {
        // TODO check shape of input
        dZ = dZ.castTo(DataType.FLOAT);
        weight.grads = cache.get("X").transpose().mmul(dZ).mul(1 / (double) currentBatchSize);
        bias.grads = dZ.sum(true, 0).muli(1 / (double) currentBatchSize);
        return dZ.mmul(weight.values.transpose());
    }
}

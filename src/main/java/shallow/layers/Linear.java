package shallow.layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.configs.LinearLayerConfig;

public class Linear extends WeightedLayer {
    int units;
    public Linear(LinearLayerConfig config) {
        super(config);
        units = config.getUnits();
    }
    // Instantiates Weight of shape [inShape[0], units] and Bias of shape [1, units]
    @Override
    public void init(long... inShape) {
        int inputSize;
        if(inShape.length == 1){
            inputSize = (int) inShape[0];
        }
        else if(inShape.length == 2){ // if inShape is a shape of type [batchSize, inputSize]
            inputSize = (int) inShape[1];
        }
        else {
            throw new IllegalArgumentException();
        }
        weight.values = Nd4j.create(DataType.FLOAT,inputSize, units);
        weightInitializer.init(inputSize, units, weight.values);
        bias.values = Nd4j.create(DataType.FLOAT, 1, units);
        biasInitializer.init(inputSize, units, bias.values);
        weight.grads = Nd4j.zerosLike(weight.values);
        bias.grads = Nd4j.zerosLike(bias.values);
    }
    // computes linear function Z = dot(W,X) + b,
    // where X.shape() = [batch_size, input_size], Z.shape() = [batch_size, output_size]
    @Override
    public INDArray forward(INDArray input) {
        // TODO check shape of input
        input = input.castTo(DataType.FLOAT);
        cache.put("X", input);
        return input.mmul(weight.values).addi(bias.values);
    }

    // computes partial derivatives chained with derivatives[0], derivatives[0].shape() = [batch_size, output_size]
    @Override
    public INDArray backward(INDArray dZ) {
        // TODO check shape of input
        dZ = dZ.castTo(DataType.FLOAT);
        long batchSize = dZ.shape()[0];
        weight.grads = cache.get("X").transpose().mmul(dZ).mul(1 / (double) batchSize);
        bias.grads = dZ.sum(true,0).muli(1 / (double)batchSize);
        return dZ.mmul(weight.values.transpose());
    }
}

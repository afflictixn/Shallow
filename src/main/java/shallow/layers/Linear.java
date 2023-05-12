package shallow.layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.configs.LinearLayerConfig;
import shallow.utils.Utils;

public class Linear extends WeightedLayer {
    LinearLayerConfig config;
    public Linear(LinearLayerConfig config) {
        super(config.input_size, config.output_size);
        this.config = config;
        initWeights();
    }
    // Instantiates Weight of shape [input_size, output_size] and Bias of shape [1, output_size]
    @Override // ToDo make separate builder for weight init
    protected void initWeights() {
        if (config.weight_initializer.equals("XavierNormal")) {
            weight.values = Nd4j.randn(config.input_size, config.output_size).muli
                    (Math.sqrt(2 / ((double) (config.input_size + config.output_size))));
        } else if (config.weight_initializer.equals("HeNormal")) {
            weight.values = Nd4j.randn(config.input_size, config.output_size).muli
                    (Math.sqrt(2 / ((double) config.input_size)));
        }
        weight.grads = Nd4j.zerosLike(weight.values);
        bias.values = Nd4j.zeros(1, config.output_size);
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
        long batch_size = dZ.shape()[0];
        weight.grads = cache.get("X").transpose().mmul(dZ).mul(1 / (double) batch_size);
        bias.grads = dZ.sum(true,0).muli(1 / (double)batch_size);
        return dZ.mmul(weight.values.transpose());
    }

}

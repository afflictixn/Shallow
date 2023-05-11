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

    @Override // ToDo make separate builder for weight init
    protected void initWeights() {
        if (config.weight_initializer.equals("XavierNormal")) {
            weight.values = Nd4j.randn(config.output_size, config.input_size).muli
                    (Math.sqrt(2 / ((double) (config.input_size + config.output_size))));
        } else if (config.weight_initializer.equals("HeNormal")) {
            weight.values = Nd4j.randn(config.output_size, config.input_size).muli
                    (Math.sqrt(2 / ((double) config.input_size)));
        }
        weight.grads = Nd4j.zerosLike(weight.values);
        bias.values = Nd4j.zeros(config.output_size, 1);
        bias.grads = Nd4j.zerosLike(bias.values);
    }

    // computes linear function Z = dot(W,X) + b, where X.shape() = [input_size, m], Z.shape() = [output_size, m]
    @Override
    public INDArray forward(INDArray input) {
        // TODO check shape of input
        input = input.castTo(DataType.FLOAT);
        cache.put("X", input);
        INDArray temp = weight.values.mmul(input).addi(bias.values);
        if(temp.isNaN().getDouble() == 1){
            System.out.println("there");
        }
        return weight.values.mmul(input).addi(bias.values);
    }

    // computes partial derivatives chained with derivatives[0], derivatives[0].shape() = [out_size, m]
    @Override
    public INDArray backward(INDArray derivatives) {
        // TODO check shape of input
        derivatives = derivatives.castTo(DataType.FLOAT);
        INDArray dZ = derivatives;
        long m = dZ.shape()[1];
        weight.grads = Utils.get().mul(dZ.mmul(cache.get("X").transpose()), (1 / (double)m));
        bias.grads = dZ.sum(true,1).muli(1 / (double)m);
        if(weight.grads.isNaN().getDouble() == 1){
            System.out.println("back");
        }
        return weight.values.transpose().mmul(dZ);
    }

}

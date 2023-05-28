package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Xavier initialization is usually used with tanh and sigmoid activation functions
public class XavierUniformInit implements WeightInitializer{
    @Override
    public void init(int in_size, int out_size, INDArray params) {
        double dev = Math.sqrt(6.0) / Math.sqrt(in_size + out_size);
        Nd4j.rand(params).muli(dev);
    }
}

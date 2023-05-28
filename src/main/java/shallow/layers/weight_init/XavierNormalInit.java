package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Xavier initialization is usually used with tanh and sigmoid activation functions
public class XavierNormalInit implements WeightInitializer {
    @Override
    public void init(int in_size, int out_size, INDArray params) {
        Nd4j.randn(params).muli(Math.sqrt(2.0 / ((double) (in_size + out_size))));
    }
}

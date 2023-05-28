package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Kaiming He initialization is usually used with ReLU activation function
public class HeNormalInit implements WeightInitializer{
    @Override
    public void init(int in_size, int out_size, INDArray params) {
        Nd4j.randn(params).muli
                (Math.sqrt(2.0 / ((double) in_size)));
    }
}

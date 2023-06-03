package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// Kaiming He initialization is usually used with ReLU activation function
public class HeUniformInit implements WeightInitializer{
    @Override
    public void init(int inSize, int outSize, INDArray params) {
        double dev = Math.sqrt(6.0 / inSize);
        Nd4j.rand(params).muli(dev);
    }
}

package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NormalInit implements WeightInitializer{
    @Override
    public void init(int inSize, int outSize, INDArray params) {
        Nd4j.rand(params).mul(0.01);
    }
}

package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface WeightInitializer {
    void init(int inSize, int outSize, INDArray params);
}

package shallow.layers.weight_init;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ConstInit implements WeightInitializer {
    private final double value;
    public ConstInit(double val){
        value = val;
    }
    @Override
    public void init(int in_size, int out_size, INDArray params) {
        params.assign(value);
    }
}

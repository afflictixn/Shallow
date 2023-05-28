package shallow.layers.configs;

import shallow.layers.weight_init.WeightInitEnum;

public class LinearLayerConfig extends WeightedLayerConfig {
    @Override
    public LinearLayerConfig inputSize(int inputSize) {
        this.inputSize = inputSize;
        return this;
    }
    @Override
    public LinearLayerConfig outputSize(int outputSize) {
        this.outputSize = outputSize;
        return this;
    }
    @Override
    public LinearLayerConfig weightInitializer(WeightInitEnum weightInit) {
        this.weightInitializer = weightInit;
        return this;
    }
}

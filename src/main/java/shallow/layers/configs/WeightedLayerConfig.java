package shallow.layers.configs;

import shallow.layers.weight_init.WeightInitEnum;

public class WeightedLayerConfig {
    public int inputSize;
    public int outputSize;
    public WeightInitEnum weightInitializer;
    public WeightInitEnum biasInitializer = WeightInitEnum.ZERO;
    public WeightedLayerConfig(){
    }

    public WeightedLayerConfig inputSize(int inputSize) {
        this.inputSize = inputSize;
        return this;
    }

    public WeightedLayerConfig outputSize(int outputSize) {
        this.outputSize = outputSize;
        return this;
    }

    public WeightedLayerConfig weightInitializer(WeightInitEnum weightInit) {
        this.weightInitializer = weightInit;
        return this;
    }

    public WeightedLayerConfig biasInitializer(WeightInitEnum biasInit) {
        this.weightInitializer = biasInit;
        return this;
    }
}

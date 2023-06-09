package shallow.layers.configs;

import shallow.layers.Linear;
import shallow.layers.weight_init.WeightInitEnum;

public class LinearLayerConfig extends WeightedLayerConfig implements Config{
    int units;
    public int getUnits() {
        return units;
    }
    public LinearLayerConfig units(int units) {
        this.units = units;
        return this;
    }
    @Override
    public LinearLayerConfig weightInitializer(WeightInitEnum weightInit) {
        this.weightInitializer = weightInit;
        return this;
    }
    @Override
    public LinearLayerConfig biasInitializer(WeightInitEnum weightInit) {
        this.biasInitializer = weightInit;
        return this;
    }
    @Override
    public Linear buildLayer() {
        return new Linear(this);
    }
}

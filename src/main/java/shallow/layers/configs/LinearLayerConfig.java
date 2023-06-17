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

    @Override
    public String getDescription(){
        return "Number of units : " + units + '\n' +
                "Weight initializer : " + weightInitializer + '\n' +
                "Bias initializer : " + biasInitializer;
    }
    @Override
    public String toString(){
        return "Linear";
    }
}

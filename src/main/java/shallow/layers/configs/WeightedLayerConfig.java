package shallow.layers.configs;

import shallow.layers.weight_init.WeightInitEnum;

public abstract class WeightedLayerConfig {
    public WeightInitEnum weightInitializer = WeightInitEnum.Normal;
    public WeightInitEnum biasInitializer = WeightInitEnum.ZEROS;
    public WeightedLayerConfig(){
    }

    public abstract WeightedLayerConfig weightInitializer(WeightInitEnum weightInit);

    public abstract WeightedLayerConfig biasInitializer(WeightInitEnum biasInit);
}

package shallow.layers.configs;

public class LinearLayerConfig extends WeightedLayerConfig {
    public String weight_initializer;
    public LinearLayerConfig(int input_size, int output_size, String weight_initializer){
        super(input_size, output_size);
        this.weight_initializer = weight_initializer;
    }
}

package shallow.layers.configs;

public class WeightedLayerConfig {
    public int input_size;
    public int output_size;
    public WeightedLayerConfig(int input_size, int output_size){
        this.input_size = input_size;
        this.output_size = output_size;
    }
}

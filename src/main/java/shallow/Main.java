package shallow;
import shallow.layers.Linear;
import shallow.layers.ReLU;
import shallow.layers.configs.LinearLayerConfig;
import shallow.losses.CategoricalCrossEntropyLoss;


public class Main {
    public static void main(String[] args) {
        int numInputs = 2;
        int numOutputs = 2;
        //Build model
        Model model = new Model();
        model.addLayer(new Linear(new LinearLayerConfig(numInputs, 50, "XavierNormal")));
        model.addLayer(new ReLU());
        model.addLayer(new Linear(new LinearLayerConfig(50, numOutputs, "XavierNormal")));
        model.addLoss(new CategoricalCrossEntropyLoss());
    }
}
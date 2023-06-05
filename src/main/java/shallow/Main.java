package shallow;
import shallow.layers.Linear;
import shallow.layers.ReLU;
import shallow.layers.configs.LinearLayerConfig;
import shallow.layers.weight_init.WeightInitEnum;
import shallow.losses.CategoricalCrossEntropyLoss;
import shallow.optimizers.Adam;


public class Main {
    public static void main(String[] args) {
        int numInputs = 2;
        int numOutputs = 2;
        //Build model
        Model model = new Model();
//        model.addLayer(new Linear(new LinearLayerConfig()
//                .inputSize(numInputs)
//                .outputSize(50)
//                .weightInitializer(WeightInitEnum.HeNormal)));
        model.addLayer(new ReLU());
//        model.addLayer(new Linear(new LinearLayerConfig()
//                .inputSize(50)
//                .outputSize(numOutputs)
//                .weightInitializer(WeightInitEnum.XavierNormal)));
        model.setLoss(new CategoricalCrossEntropyLoss());
        model.setOptimizer(new Adam());
    }
}
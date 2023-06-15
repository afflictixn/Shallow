package shallow.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import shallow.layers.WeightedLayer;

import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public abstract class BaseOptimizer {
    List<WeightedLayer> layers; // layers for which weight update is performed
    double currentLambda = 0.0;
    double currentLearningRate;
    INDArray[] currentGrads = new INDArray[2]; // gradients of weight and bias of a layer to be updated, respectively
    INDArray[] currentValues = new INDArray[2]; // currentGrads[0] is weight of a layer to be updated, currentGrads[1] is bias
    BiConsumer<Integer, Integer> updateFunction;
    void regularizeL2(int index) {
        INDArray weight = currentValues[0];
        weight.subi(weight.mul(currentLambda / layers.get(index).getCurrentBatchSize() ));
    }
    public void addRegularizationL2(double lambda) {
        currentLambda = lambda;
        if(currentLambda != 0.0) {
            BiConsumer<Integer, Integer> oldUpdate = updateFunction;
            updateFunction = (index, iteration)->{
                oldUpdate.accept(index, iteration);
                regularizeL2(index);
            };
        }
    }
    public abstract void init(List<WeightedLayer> layers);

    public void updateWeights(double learningRate, int curIteration) {
        currentLearningRate = learningRate;
        int index = 0;
        for (WeightedLayer layer : layers) {
            currentGrads[0] = layer.getWeightGrads();
            currentGrads[1] = layer.getBiasGrads();
            currentValues[0] = layer.getWeightValues();
            currentValues[1] = layer.getBiasValues();
            updateFunction.accept(index, curIteration);
            ++index;
        }
    }
}

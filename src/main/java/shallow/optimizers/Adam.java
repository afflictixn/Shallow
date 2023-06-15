package shallow.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.WeightedLayer;
import shallow.utils.Utils;

import java.util.ArrayList;
import java.util.List;

public class Adam extends BaseOptimizer {
    double beta1 = 0.9; // Exponential decay hyperparameter for the first moment estimates
    double beta2 = 0.999; // Exponential decay hyperparameter for the second moment estimates
    List<List<INDArray>> waGrads; // stores exponentially weighted average of past gradients of every layer
    List<List<INDArray>> waSqGrads; // stores exponentially weighted average of the squares of the past gradients of every layer
    void updateLayer(int index, int iteration) {
        for (int j = 0; j < currentGrads.length; ++j) {
            // compute new weighted average (muli means we multiply in place, so we don't need to set the new value)
            waGrads.get(index).get(j).muli(beta1).addi(currentGrads[j].mul(1 - beta1));
            // compute weighted average with bias correction
            INDArray waGradCorrected = waGrads.get(index).get(j).div(1 - Math.pow(beta1, iteration));
            // compute new weighted average for squares of gradients
            waSqGrads.get(index).get(j).muli(beta2).addi(currentGrads[j].mul(currentGrads[j]).muli(1 - beta2));
            // introduce bias correction
            INDArray waSqGradCorrected = waSqGrads.get(index).get(j).div(1 - Math.pow(beta2, iteration));
            // update weights, epsilon is added for numeric stability
            currentValues[j].subi(waGradCorrected.divi(Utils.get().sqrt(waSqGradCorrected).addi(Utils.epsilon8)).muli(currentLearningRate));
        }
    }
    @Override
    public void init(List<WeightedLayer> layers) {
        this.layers = layers;
        waGrads = new ArrayList<>();
        waSqGrads = new ArrayList<>();
        for (WeightedLayer layer : layers) {
            waGrads.add(new ArrayList<>());
            waSqGrads.add(new ArrayList<>());
            waGrads.get(waGrads.size() - 1).add(Nd4j.zerosLike(layer.getWeightGrads()));
            waGrads.get(waGrads.size() - 1).add(Nd4j.zerosLike(layer.getBiasGrads()));
            waSqGrads.get(waSqGrads.size() - 1).add(Nd4j.zerosLike(layer.getWeightGrads()));
            waSqGrads.get(waSqGrads.size() - 1).add(Nd4j.zerosLike(layer.getBiasGrads()));
        }
    }

    public Adam() {
        updateFunction = this::updateLayer;
    }

    public Adam(double beta1, double beta2) {
        this();
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

}

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
    }

    public Adam(double beta1, double beta2) {
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public void updateWeights(double learning_rate, int cur_iteration) {
        INDArray[] grads = new INDArray[2];
        INDArray[] values = new INDArray[2];
        int index = 0;
        for (WeightedLayer layer : layers) {
            grads[0] = layer.getWeightGrads();
            grads[1] = layer.getBiasGrads();
            values[0] = layer.getWeightValues();
            values[1] = layer.getBiasValues();
            for (int j = 0; j < grads.length; ++j) {
                // compute new weighted average (muli means we multiply in place, so we don't need to set the new value)
                waGrads.get(index).get(j).muli(beta1).addi(grads[j].mul(1 - beta1));
                // compute weighted average with bias correction
                INDArray waGradCorrected = waGrads.get(index).get(j).div(1 - Math.pow(beta1, cur_iteration));
                // compute new weighted average for squares of gradients
                waSqGrads.get(index).get(j).muli(beta2).addi(grads[j].mul(grads[j]).muli(1 - beta2));
                // introduce bias correction
                INDArray waSqGradCorrected = waSqGrads.get(index).get(j).div(1 - Math.pow(beta2, cur_iteration));
                // update weights, epsilon is added for numeric stability
                values[j].subi(waGradCorrected.divi(Utils.get().sqrt(waSqGradCorrected).addi(Utils.epsilon8)).muli(learning_rate));
            }
            ++index;
        }
    }

}

package shallow.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.BaseLayer;
import shallow.layers.WeightedLayer;

import java.util.ArrayList;
import java.util.List;

public class SGD extends BaseOptimizer {
    double momentum = 0.0;
    List<List<INDArray>> velocity = null;

    @Override
    public void init(List<WeightedLayer> layers) {
        this.layers = layers;
        velocity = new ArrayList<>();
        for (WeightedLayer layer : layers) {
            velocity.add(new ArrayList<>());
            velocity.get(velocity.size() - 1).add(Nd4j.zerosLike(layer.getWeightGrads()));
            velocity.get(velocity.size() - 1).add(Nd4j.zerosLike(layer.getBiasGrads()));
        }
    }

    public SGD() {

    }

    public SGD(double momentum) {
        this.momentum = momentum;
    }

    @Override
    public void updateWeights(double learning_rate, int cur_iteration) {
        INDArray[] grads = new INDArray[2];
        INDArray[] values = new INDArray[2];
        int index = 0;
        if (momentum != 0.0) {
            for (WeightedLayer layer : layers) {
                grads[0] = layer.getWeightGrads();
                grads[1] = layer.getBiasGrads();
                values[0] = layer.getWeightValues();
                values[1] = layer.getBiasValues();
                for (int j = 0; j < grads.length; ++j) {
                    velocity.get(index).get(j).muli(momentum).subi(grads[j].mul(learning_rate));
                    values[j].addi(velocity.get(index).get(j));
                }
                ++index;

            }
        } else {
            for (WeightedLayer layer : layers) {
                grads[0] = layer.getWeightGrads();
                grads[1] = layer.getBiasGrads();
                values[0] = layer.getWeightValues();
                values[1] = layer.getBiasValues();
                for (int j = 0; j < grads.length; ++j) {
                    values[j].subi(grads[j].mul(learning_rate));
                }
                ++index;
            }
        }
    }
}

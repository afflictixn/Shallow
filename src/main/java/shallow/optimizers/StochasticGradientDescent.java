package shallow.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.WeightedLayer;

import java.util.ArrayList;
import java.util.List;

public class StochasticGradientDescent extends BaseOptimizer {
    double momentum = 0.0;
    List<List<INDArray>> velocity = null;
    void updateLayerMomentum(int index) {
        for (int j = 0; j < currentGrads.length; ++j) {
            velocity.get(index).get(j).muli(momentum).subi(currentGrads[j].mul(currentLearningRate));
            currentValues[j].addi(velocity.get(index).get(j));
        }
    }
    void updateLayerSimple() {
        for (int j = 0; j < currentGrads.length; ++j) {
            currentValues[j].subi(currentGrads[j].mul(currentLearningRate));
        }
    }

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

    public StochasticGradientDescent() {
        updateFunction = (index, iteration) -> {
            updateLayerSimple();
        };
    }

    public StochasticGradientDescent(double momentum) {
        this.momentum = momentum;
        updateFunction = (momentum == 0.0) ? (index, iteration) -> updateLayerSimple() : (index, iteration) -> updateLayerMomentum(index);
    }

}

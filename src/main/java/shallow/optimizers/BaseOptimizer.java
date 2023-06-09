package shallow.optimizers;

import shallow.layers.WeightedLayer;

import java.util.List;

public abstract class BaseOptimizer {
    List<WeightedLayer> layers; // layers for which weight update is performed

    public abstract void init(List<WeightedLayer> layers);

    public abstract void updateWeights(double learning_rate, int cur_iteration);
}

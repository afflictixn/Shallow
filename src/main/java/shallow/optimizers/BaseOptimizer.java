package shallow.optimizers;

import shallow.layers.BaseLayer;

import java.util.List;

public abstract class BaseOptimizer {
    List<BaseLayer> layers; // layers for which weight update is performed
    public abstract void init(List<BaseLayer> layers);
    public abstract void updateWeights(double learning_rate, int cur_iteration);
}

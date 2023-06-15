package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import shallow.layers.configs.WeightedLayerConfig;
import shallow.layers.weight_init.WeightInitializer;

class Weight {
    public INDArray values;
    public INDArray grads;

    public void setValues(INDArray values) {
        this.values = values;
    }

    public void setGrads(INDArray grads) {
        this.grads = grads;
    }
}

class Bias {
    public INDArray values;
    public INDArray grads;

    public void setValues(INDArray values) {
        this.values = values;
    }

    public void setGrads(INDArray grads) {
        this.grads = grads;
    }
}

public abstract class WeightedLayer extends BaseLayer {
    long currentBatchSize;
    Weight weight = new Weight();
    Bias bias = new Bias();
    WeightInitializer weightInitializer;
    WeightInitializer biasInitializer;
    public WeightedLayer(WeightedLayerConfig config) {
        this.weightInitializer = config.weightInitializer.getWeightInitializer();
        this.biasInitializer = config.biasInitializer.getWeightInitializer();
    }
    public long getCurrentBatchSize() {
        return currentBatchSize;
    }
    public INDArray getWeightValues() {
        return weight.values;
    }
    public INDArray getWeightGrads() {
        return weight.grads;
    }
    public INDArray getBiasValues() {
        return bias.values;
    }
    public INDArray getBiasGrads() {
        return bias.grads;
    }
}

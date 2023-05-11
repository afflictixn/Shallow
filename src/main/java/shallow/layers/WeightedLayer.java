package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

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

public abstract class WeightedLayer extends BaseLayer{
    Weight weight = new Weight();
    Bias bias = new Bias();
    int input_size, output_size;
    public WeightedLayer(int input_size, int output_size) {
        this.input_size = input_size;
        this.output_size = output_size;
    }
    protected abstract void initWeights();
    public int getInputSize(){
        return input_size;
    }
    public int getOutputSize(){
        return output_size;
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

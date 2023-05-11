package shallow;

import org.nd4j.linalg.api.ndarray.INDArray;
import shallow.layers.BaseLayer;
import shallow.layers.WeightedLayer;
import shallow.losses.BaseLoss;
import shallow.optimizers.Adam;
import shallow.optimizers.BaseOptimizer;

import java.util.ArrayList;
import java.util.List;

public class Model {
    List<BaseLayer> layers;
    BaseLoss loss;
    int last_layer_size;
    public Model() {
        layers = new ArrayList<>();
    }

    public void addLayer(BaseLayer layer) {
        if (!validLayerCheck(layer)) {
            throw new IllegalArgumentException();
        }
        layers.add(layer);
    }
    public void addLoss(BaseLoss loss) {
        this.loss = loss;
    }

    // sequential forward pass of a model
    public INDArray forwardPass(INDArray X) {
        for (BaseLayer layer : layers) {
            X = layer.forward(X);
        }
        return X;
    }

    public double computeLoss(INDArray X, INDArray Y) {
        return loss.forward(X, Y).getDouble();
    }

    public void backwardPass() {
        INDArray D = loss.backward();
        for (int i = layers.size() - 1; i >= 0; --i) {
            D = layers.get(i).backward(D);
        }
    }
    // ToDo rewrite fit to allow mini-batch training
    public void fit(INDArray X, INDArray Y, double learning_rate, int num_iterations) {
        BaseOptimizer  optimizer = new Adam(layers);
        INDArray result = null;
        for(int i = 1; i <= num_iterations; ++i){
            result = forwardPass(X);
            double value = computeLoss(result, Y);
            if(i % 50 == 1 || i == num_iterations) {
                System.out.println(value);
            }
            backwardPass();
            optimizer.updateWeights(learning_rate, i);
        }
    }

    private boolean validLayerCheck(BaseLayer layer) {
        boolean ok = true;
        if (layer instanceof WeightedLayer weighted) {
            ok = (last_layer_size == 0) || last_layer_size == weighted.getInputSize();
            last_layer_size = weighted.getOutputSize();
        }
        return ok;
    }
}

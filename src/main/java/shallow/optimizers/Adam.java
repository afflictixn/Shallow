package shallow.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.BaseLayer;
import shallow.layers.WeightedLayer;
import shallow.utils.Utils;

import java.util.ArrayList;
import java.util.List;

public class Adam extends BaseOptimizer {
    double beta1 = 0.9; // Exponential decay hyperparameter for the first moment estimates
    double beta2 = 0.999; // Exponential decay hyperparameter for the second moment estimates
    List<List<INDArray>> wa_grads; // stores exponentially weighted average of past gradients of every layer
    List<List<INDArray>> wa_sq_grads; // stores exponentially weighted average of the squares of the past gradients of every layer
    public Adam(List<BaseLayer> layers) {
        this.layers = layers;
        wa_grads = new ArrayList<>();
        wa_sq_grads = new ArrayList<>();
        for(BaseLayer layer : layers){
            if(layer instanceof WeightedLayer weighted) {
                wa_grads.add(new ArrayList<>());
                wa_sq_grads.add(new ArrayList<>());
                wa_grads.get(wa_grads.size() - 1).add(Nd4j.zerosLike(weighted.getWeightGrads()));
                wa_grads.get(wa_grads.size() - 1).add(Nd4j.zerosLike(weighted.getBiasGrads()));
                wa_sq_grads.get(wa_sq_grads.size() - 1).add(Nd4j.zerosLike(weighted.getWeightGrads()));
                wa_sq_grads.get(wa_sq_grads.size() - 1).add(Nd4j.zerosLike(weighted.getBiasGrads()));
            }
        }
    }

    public Adam(List<BaseLayer> layers, double beta1, double beta2) {
        this(layers);
        this.beta1 = beta1;
        this.beta2 = beta2;
    }
    @Override
    public void updateWeights(double learning_rate, int cur_iteration) {
        INDArray [] grads = new INDArray[2]; INDArray [] values = new INDArray[2];
        int index = 0;
        for (BaseLayer layer : layers) {
            if (layer instanceof WeightedLayer weighted) {
                 grads[0] = weighted.getWeightGrads(); grads[1] = weighted.getBiasGrads();
                 values[0] = weighted.getWeightValues(); values[1] = weighted.getBiasValues();
                 for(int j = 0; j < grads.length; ++j) {
                     // compute new weighted average (muli means we multiply in place, so we don't need to set the new value)
                     wa_grads.get(index).get(j).muli(beta1).addi(grads[j].mul(1 - beta1));
                     // compute weighted average with bias correction
                     INDArray wa_grad_corrected = wa_grads.get(index).get(j).div(1 - Math.pow(beta1, cur_iteration));
                     // compute new weighted average for squares of gradients
                     wa_sq_grads.get(index).get(j).muli(beta2).addi(grads[j].mul(grads[j]).muli(1 - beta2));
                     // introduce bias correction
                     INDArray wa_sq_grad_corrected = wa_sq_grads.get(index).get(j).div(1 - Math.pow(beta2, cur_iteration));
                     // update weights, epsilon is added for numeric stability
                     values[j].subi(wa_grad_corrected.divi(Utils.get().sqrt(wa_sq_grad_corrected).addi(Utils.epsilon8)).muli(learning_rate));
                 }
                 ++index;
            }
        }
    }
}

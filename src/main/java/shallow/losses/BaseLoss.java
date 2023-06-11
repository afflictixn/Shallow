package shallow.losses;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

public abstract class BaseLoss {
    // local gradients of this function
    Map<String, INDArray> grads;
    Map<String, INDArray> cache;

    public BaseLoss() {
        grads = new HashMap<>();
        cache = new HashMap<>();
    }

    // computes the loss function of X with respect to target Y
    public abstract INDArray forward(INDArray X, INDArray Y);

    public abstract INDArray backward();
    public abstract INDArray getActivation();
    public Map<String, INDArray> getGrads() {
        return grads;
    }

}

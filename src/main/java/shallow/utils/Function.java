package shallow.utils;

import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Function {
    // local gradients of this function
    Map<String, INDArray> grads;
    Map<String, INDArray> cache;
    public Function(){
        grads = new HashMap<>();
        cache = new HashMap<>();
    }
    public abstract INDArray forward(INDArray input);
    public abstract INDArray backward(INDArray input);

    public INDArray call(INDArray input) throws Exception {
        INDArray output = forward(input);
        grads = getGrads(input);
        return output;
    }
    // computes partial derivatives of input with respect to this function
    public abstract Map<String, INDArray> getGrads(INDArray input) throws Exception;
}


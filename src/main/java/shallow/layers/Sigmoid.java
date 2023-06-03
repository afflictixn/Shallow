package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.utils.Utils;

public class Sigmoid extends BaseLayer{
    public Sigmoid() {
    }
    // computes element-wise sigmoid function f = 1 / (1 + e^{-z}) for every entry z of input
    @Override
    public INDArray forward(INDArray input) {
        INDArray activation = Nd4j.onesLike(input).divi(Utils.get().exp(input.neg()).addi(1));
        INDArray localGrad = activation.mul(Nd4j.ones(activation.shape()).subi(activation));
        cache.put("ldX", localGrad);
        return activation;
    }
    @Override
    public INDArray backward(INDArray derivatives) {
        return derivatives.mul(cache.get("ldX"));
    }
}

package shallow.losses;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.utils.Utils;

public class BCELoss extends BaseLoss {

    // computes binary cross entropy loss of array of probabilities X with respect to labels 1, 0 in Y
    // X and Y are of shape [1, num_samples]
    @Override
    public INDArray forward(INDArray X, INDArray Y) {
        X = X.castTo(DataType.FLOAT);
        Y = Y.castTo(DataType.FLOAT);
        cache.put("X", X);
        cache.put("Y", Y);
        // clip X in order to avoid log(0)
        X = Utils.get().clipByValue(X, Utils.epsilon8, 1. - Utils.epsilon4);
        // calculate - 1/m sum (Y*log(X) + (1 - Y)*log(1 - X)), where sum goes through all m samples
        INDArray cost = Y.mul(Utils.get().log(X.dup())).addi(
                        Nd4j.onesLike(Y).subi(Y).mul(Utils.get().log(Nd4j.onesLike(X).subi(X)))).sum(1)
                .mul(-1.0 / X.shape()[1]);
        return cost;
    }

    // derivative of BCE, i.e. (X - Y) / X(1 - X), where division and multiplication are pairwise
    @Override
    public INDArray backward() {
        INDArray X = cache.get("X");
        INDArray Y = cache.get("Y");
        // epsilon is added for numerical stability
        grads.put("dX", X.sub(Y).divi(X.mul(Nd4j.onesLike(X).subi(X)).addi(Utils.epsilon12)));
        return grads.get("dX");
    }
}

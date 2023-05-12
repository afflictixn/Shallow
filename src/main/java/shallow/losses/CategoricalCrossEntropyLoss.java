package shallow.losses;

import org.nd4j.linalg.api.ndarray.INDArray;
import shallow.utils.Utils;

// Use CategoricalCrossEntropyLoss function when there are two or more label classes.
// Softmax is integrated in this loss function
// It's expected that labels are provided in a one_hot representation.
public class CategoricalCrossEntropyLoss extends BaseLoss {
    // X - logits of shape [batch_size, num_classes]
    // Y - labels for each example of shape [batch_size, num_classes]
    @Override
    public INDArray forward(INDArray X, INDArray Y) {
        cache.put("Y", Y);
        // for numerical stability, for each sample we apply not softmax(x), but softmax(x - max(x_i))
        // where x is sample vector and max(x_i) is a maximum value along this vector, result is analytically the same
        INDArray corrected = Utils.get().exp(X.sub(X.max(true, 1)));
        INDArray activation = corrected.div(corrected.sum(true,1));
        cache.put("A", activation);
        INDArray cost = Y.mul(Utils.get().log(activation.add(Utils.epsilon8))).sum();
        return cost;
    }
    // derivative of CrossEntropyLoss with Softmax, i.e. Softmax(X) - Y
    @Override
    public INDArray backward() {
        INDArray A = cache.get("A");
        INDArray Y = cache.get("Y");
        return A.sub(Y);
    }
}

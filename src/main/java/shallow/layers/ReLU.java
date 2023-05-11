package shallow.layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.utils.Utils;

public class ReLU extends BaseLayer{
    @Override
    public INDArray forward(INDArray input) {
        cache.put("Z", Utils.get().max(input, Nd4j.zerosLike(input)));
        return cache.get("Z");
    }

    @Override
    public INDArray backward(INDArray derivatives) {
        return derivatives.mul(cache.get("Z").neq(0).castTo(DataType.FLOAT));
    }
}

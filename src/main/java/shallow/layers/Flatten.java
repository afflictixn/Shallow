package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Flatten extends BaseLayer {
    long[] shape = null;
    // takes as input tensor of shape [batch_size, channels, height, width],
    // outputs tensor of shape [batch_size, channels * height * width]
    @Override
    public INDArray forward(INDArray input) {
        if(shape == null) {
            shape = input.shape().clone();
        }
        return input.reshape(input.shape()[0], input.shape()[1] * input.shape()[2] * input.shape()[3]);
    }

    @Override
    public INDArray backward(INDArray derivatives) {
        return derivatives.reshape(derivatives.shape()[0], shape[1], shape[2], shape[3]);
    }
}

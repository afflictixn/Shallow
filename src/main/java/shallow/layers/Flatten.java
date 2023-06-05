package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Flatten extends BaseLayer implements ShapeChangingLayer{
    long height, width, channels, product;
    // takes as input tensor of shape [batch_size, channels, height, width],
    // outputs tensor of shape [batch_size, channels * height * width]
    @Override
    public INDArray forward(INDArray input) {
        return input.reshape(input.shape()[0], product);
    }
    @Override
    public INDArray backward(INDArray derivatives) {
        return derivatives.reshape(derivatives.shape()[0], height, width, channels);
    }
    @Override
    public void init(long... inShape) {
        if (inShape.length != 3 && inShape.length != 4) {
            throw new IllegalArgumentException();
        }
        int offset = (inShape.length == 3) ? 0 : 1;
        height = inShape[offset];
        width = inShape[offset + 1];
        channels = inShape[offset + 2];
        product = height * width * channels;
    }
    @Override
    public long[] getOutputShape() {
        return new long[]{product};
    }
}

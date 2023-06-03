package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.configs.MaxPool2dConfig;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class MaxPool2d extends BaseLayer {
    int[] kernelSize, stride; // [0] for height, [1] for width
    long[] shapeBefore; // [batchSize, Height, Width, Channels]
    long outputHeight, outputWidth;

    public MaxPool2d() {
        kernelSize = new int[]{2, 2};
        stride = new int[]{2, 2};
    }

    public MaxPool2d(MaxPool2dConfig config) {
        kernelSize = config.getKernelSize();
        stride = config.getStrides();
    }

    @Override
    public INDArray forward(INDArray input) {
        int batchSize = (int) input.shape()[0];
        shapeBefore = input.shape().clone();
        outputHeight = (int) (shapeBefore[1] - kernelSize[0]) / stride[0] + 1;
        outputWidth = (int) (shapeBefore[2] - kernelSize[1]) / stride[1] + 1;
        INDArray output = Nd4j.zeros(batchSize, outputHeight, outputWidth, shapeBefore[3]);
        INDArray mask = Nd4j.zerosLike(input);
        // apply pooling
        int verticalStart, verticalEnd, horizontalStart, horizontalEnd;
        for (int i = 0; i < outputHeight; ++i) {
            verticalStart = stride[0] * i;
            verticalEnd = verticalStart + kernelSize[0];
            for (int j = 0; j < outputWidth; ++j) {
                horizontalStart = stride[1] * j;
                horizontalEnd = horizontalStart + kernelSize[1];

                INDArray inputSlice = input.get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd));
                output.get(all(), point(i), point(j)).assign(inputSlice.max(1, 2));
                // save mask of maximum elements to use in backward step
                // TODO introduce training and prediction modes
                INDArray broadcasted = inputSlice.max(true, 1, 2).broadcast(inputSlice.shape());

                INDArray maxMask = inputSlice.eq(broadcasted);
                mask.get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd)).assign(maxMask);
            }
        }
        cache.put("mask", mask);
        return output;
    }

    @Override
    public INDArray backward(INDArray dZ) {
        INDArray mask = cache.get("mask");
        int batchSize = (int) dZ.shape()[0];
        INDArray dInput = Nd4j.zeros(batchSize, shapeBefore[1], shapeBefore[2], shapeBefore[3]);

        int verticalStart, verticalEnd, horizontalStart, horizontalEnd;
        for (int i = 0; i < outputHeight; ++i) {
            verticalStart = stride[0] * i;
            verticalEnd = verticalStart + kernelSize[0];
            for (int j = 0; j < outputWidth; ++j) {
                horizontalStart = stride[1] * j;
                horizontalEnd = horizontalStart + kernelSize[1];

                INDArray dInputSlice = dInput.get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd));
                dInputSlice.addi(
                        dZ.get(all(), interval(i, i + 1), interval(j, j + 1))
                                .mul(mask.get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd))));
            }
        }
        return dInput;
    }
}

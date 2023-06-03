package shallow.layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import shallow.layers.configs.Conv2dConfig;
import shallow.layers.configs.PaddingType;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class Conv2d extends WeightedLayer {
    PaddingType paddingType;
    int[] kernelSize, stride; // [0] for height, [1] for width
    int[][] padding;
    long outputHeight, outputWidth, outputChannels;
    long inputHeight, inputWidth, inputChannels;

    public Conv2d(Conv2dConfig config) {
        super(config);
        kernelSize = config.getKernelSize();
        stride = config.getStrides();
        outputChannels = config.getFilters();
        paddingType = config.getPaddingType();
        if (paddingType.equals(PaddingType.NONE)) {
            padding = new int[][]{{0, 0}, {config.getPadding()[0], config.getPadding()[0]},
                    {config.getPadding()[1], config.getPadding()[1]}, {0, 0}};
        }
    }

    // inShape is either of form [inputHeight, inputWidth, inputChannels] or
    // [bathSize, inputHeight, inputWidth, inputChannels]
    @Override
    public void init(long... inShape) {
        if (inShape.length == 4) {
            inputHeight = inShape[1];
            inputWidth = inShape[2];
            inputChannels = inShape[3];
        } else if (inShape.length == 3) {
            inputHeight = inShape[0];
            inputWidth = inShape[1];
            inputChannels = inShape[2];
        } else {
            throw new IllegalArgumentException();
        }
        // process padding type
        if (paddingType.equals(PaddingType.VALID)) {
            padding = new int[][]{{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        } else if (paddingType.equals(PaddingType.SAME)) {
            double totalPaddingHeight = (inputHeight - 1) * stride[0] - inputHeight + kernelSize[0];
            int topHeight = (int) Math.floor(totalPaddingHeight / 2);
            int bottomHeight = (int) Math.ceil(totalPaddingHeight / 2);
            double totalPaddingWidth = (inputWidth - 1) * stride[1] - inputWidth + kernelSize[1];
            int leftWidth = (int) Math.floor(totalPaddingWidth / 2);
            int rightWidth = (int) Math.ceil(totalPaddingWidth / 2);
            padding = new int[][]{{0, 0}, {topHeight, bottomHeight}, {leftWidth, rightWidth}, {0, 0}};
        }
        outputHeight = (int) (inputHeight + padding[1][0] + padding[1][1] - kernelSize[0]) / stride[0] + 1;
        outputWidth = (int) (inputWidth + padding[2][0] + padding[2][1] - kernelSize[1]) / stride[1] + 1;
        int inSize = (int) (inputChannels * kernelSize[0] * kernelSize[1]);
        int outSize = (int) ((outputChannels * kernelSize[0] * kernelSize[1]) / (stride[0] * stride[1]));
        // initialize weights and grads
        weight.values = Nd4j.create(kernelSize[0], kernelSize[1], inputChannels, outputChannels);
        weightInitializer.init(inSize, outSize, weight.values);
        weight.grads = Nd4j.zerosLike(weight.values);
        bias.values = Nd4j.create(1, 1, 1, outputChannels);
        biasInitializer.init(inSize, outSize, bias.values);
        bias.grads = Nd4j.zerosLike(bias.values);
    }

    // input of shape [batchSize, inputHeight, inputWidth, inputChannels]
    // weight tensors are of shape [kernelSize, kernelSize, in_channels, out_channels]
    @Override
    public INDArray forward(INDArray input) {
        if(!input.dataType().equals(DataType.FLOAT)){
            input.castTo(DataType.FLOAT);
        }
//        weight.values = Nd4j.ones()
        long batchSize = input.shape()[0];
        INDArray inputPadded = Nd4j.pad(input, padding);
        // ouput of shape [batch, out_height, out_width, out_channels]
        INDArray output = Nd4j.zeros(batchSize, outputHeight, outputWidth, outputChannels);
        int verticalStart, verticalEnd, horizontalStart, horizontalEnd;
        for (int i = 0; i < outputHeight; ++i) {
            verticalStart = stride[0] * i;
            verticalEnd = verticalStart + kernelSize[0];
            for (int j = 0; j < outputWidth; ++j) {
                horizontalStart = stride[1] * j;
                horizontalEnd = horizontalStart + kernelSize[1];
                INDArray slicedInput =
                        inputPadded.get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd), all());
                INDArray newSlice = Nd4j.expandDims(slicedInput, 4);
                output.put(new INDArrayIndex[]{all(), point(i), point(j), all()},
                        newSlice.mul(weight.values.get(newAxis(), all(), all(), all())).sum(1, 2, 3));
//                for(int kk = 0; kk < 2; ++kk){
//                    System.out.println("This:");
//                    System.out.println(output.get(point(kk), point(i), point(j)));
//                }
            }
        }
        output.addi(bias.values);
        cache.put("inputPadded", inputPadded);
        return output;
    }

    @Override
    public INDArray backward(INDArray dZ) {
        if(!dZ.dataType().equals(DataType.FLOAT)){
            dZ.castTo(DataType.FLOAT);
        }
        int batchSize = (int) dZ.shape()[0];
        INDArray dInputPadded = Nd4j.zerosLike(cache.get("inputPadded"));
        INDArray InputPaddedExpanded = Nd4j.expandDims(cache.get("inputPadded"), 4);
        INDArray dZExpanded = Nd4j.expandDims(dZ, 3);
        INDArray WExpanded = Nd4j.expandDims(weight.values, 0);
        // Calculate db
        INDArray db = dZ.sum(0, 1, 2).divi(batchSize);

        int verticalStart, verticalEnd, horizontalStart, horizontalEnd;
        for (int i = 0; i < outputHeight; ++i) {
            verticalStart = stride[0] * i;
            verticalEnd = verticalStart + kernelSize[0];
            for (int j = 0; j < outputWidth; ++j) {
                horizontalStart = stride[1] * j;
                horizontalEnd = horizontalStart + kernelSize[1];

                // Calculating derivatives of Input
                INDArray dZExpandedSlice = dZExpanded.get(all(), interval(i, i + 1), interval(j, j + 1));

                dInputPadded
                        .get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd))
                        .addi(WExpanded.mul(dZExpandedSlice).sum(4));
                // Calculating derivatives of Weight
                weight.grads.addi(InputPaddedExpanded
                        .get(all(), interval(verticalStart, verticalEnd), interval(horizontalStart, horizontalEnd))
                        .mul(dZExpandedSlice).sum(0));

            }
        }

        weight.grads.divi(batchSize);
        // select derivatives of original tensor without padding

        INDArray dInput = dInputPadded.get(
                all(),
                interval(padding[1][0], dInputPadded.size(1) - padding[1][1]),
                interval(padding[2][0], dInputPadded.size(2) - padding[2][1]));

        return dInput;
    }
}

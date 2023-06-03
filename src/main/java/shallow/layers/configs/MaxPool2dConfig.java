package shallow.layers.configs;

import shallow.layers.weight_init.WeightInitEnum;

public class MaxPool2dConfig {
    int[] kernelSize, strides;
    public int[] getKernelSize() {
        return kernelSize;
    }

    public int[] getStrides() {
        return strides;
    }
    // TODO check input length
    public MaxPool2dConfig kernelSize(int... kernelSize) {
        this.kernelSize = kernelSize;
        return this;
    }
    public MaxPool2dConfig strides(int... strides) {
        this.strides = strides;
        return this;
    }

}

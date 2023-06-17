package shallow.layers.configs;

import shallow.layers.BaseLayer;
import shallow.layers.MaxPool2d;
import shallow.layers.weight_init.WeightInitEnum;

public class MaxPool2dConfig implements Config {
    int[] kernelSize = new int[] {2, 2}, strides = new int[] {2, 2};
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
    @Override
    public MaxPool2d buildLayer() {
        return new MaxPool2d(this);
    }

    @Override
    public String getDescription(){
        return "Kernel height : " + kernelSize[0] + '\n' +
                "Kernel width : " + kernelSize[1] + '\n' +
                "Strides height : " + strides[0] + '\n' +
                "Strides width : " + strides[1];
    }
    @Override
    public String toString(){
        return "MaxPool2d";
    }
}

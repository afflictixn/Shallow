package shallow.layers.configs;

import shallow.layers.Conv2d;
import shallow.layers.weight_init.WeightInitEnum;

public class Conv2dConfig extends WeightedLayerConfig implements Config{
    int filters = 3;
    int[] kernelSize = new int[] {3, 3}, strides = new int[] {1, 1}, padding = null;
    PaddingType paddingType = PaddingType.SAME;
    public int getFilters() {
        return filters;
    }
    public int[] getKernelSize() {
        return kernelSize;
    }

    public int[] getStrides() {
        return strides;
    }

    public int[] getPadding() {
        return padding;
    }

    public PaddingType getPaddingType() {
        return paddingType;
    }
    // TODO check input length
    public Conv2dConfig kernelSize(int... kernelSize) {
        this.kernelSize = kernelSize;
        return this;
    }
    public Conv2dConfig strides(int... strides) {
        this.strides = strides;
        return this;
    }
    public Conv2dConfig padding(int... padding) {
        this.padding = padding;
        return this;
    }
    public Conv2dConfig paddingType(PaddingType paddingType) {
        this.paddingType = paddingType;
        return this;
    }
    public Conv2dConfig filters(int filters) {
        this.filters = filters;
        return this;
    }
    @Override
    public Conv2dConfig weightInitializer(WeightInitEnum weightInit) {
        this.weightInitializer = weightInit;
        return this;
    }
    @Override
    public Conv2dConfig biasInitializer(WeightInitEnum biasInit) {
        this.biasInitializer = biasInit;
        return this;
    }

    @Override
    public Conv2d buildLayer() {
        return new Conv2d(this);
    }

    @Override
    public String getDescription(){
        return "Number of filters : " + filters + '\n' +
                "Kernel height : " + kernelSize[0] + '\n' +
                "Kernel width : " + kernelSize[1] + '\n' +
                "Strides height : " + strides[0] + '\n' +
                "Strides width : " + strides[1] + '\n' +
                "Padding type : " + paddingType;
    }
    @Override
    public String toString(){
        return "Conv2d";
    }
}

package home.gui;

import java.util.ArrayList;
import java.util.List;

import shallow.layers.Flatten;
import shallow.layers.ReLU;
import shallow.layers.Sigmoid;
import shallow.layers.configs.*;
import shallow.layers.weight_init.WeightInitEnum;
import shallow.losses.LossEnum;
import shallow.optimizers.OptimizerEnum;

public class Connector {
    DatasetEnum datasetEnum;
    List<Config> configs;
    HyperParametersInfo hyperParametersInfo = new HyperParametersInfo();
    OptimizerEnum optimizerEnum;
    LossEnum lossEnum;
    public void setDatasetEnum(DatasetEnum dataset) {
        datasetEnum = dataset;
    };
    public void setLossEnum(LossEnum lossEnum) {
        this.lossEnum = lossEnum;
    }
    public void setOptimizerEnum(OptimizerEnum optimizerEnum) {
        this.optimizerEnum = optimizerEnum;
    }
    public void processLinear(int nUnits, WeightInitEnum weightInit, WeightInitEnum weightBias) {
        configs.add(new LinearLayerConfig().units(nUnits).weightInitializer(weightInit).biasInitializer(weightBias));
    }

    public void processReLU() {
        configs.add(ReLU::new);
    }

    public void processSigmoid() {
        configs.add(Sigmoid::new);
    }
    public void processConv2d(int filters,
                              int kernelHeight, int kernelWidth,
                              int stridesHeight, int stridesWidth,
                              PaddingType paddingType) {
        configs.add(new Conv2dConfig().filters(filters)
                .kernelSize(kernelHeight, kernelWidth)
                .strides(stridesHeight, stridesWidth)
                .paddingType(paddingType)
        );
    }
    public void processMaxPool2d(int kernelHeight, int kernelWidth,
                                 int stridesHeight, int stridesWidth) {
        configs.add(new MaxPool2dConfig().kernelSize(kernelHeight, kernelWidth).strides(stridesHeight, stridesWidth));
    }
    public void processFlatten() {
        configs.add(Flatten::new);
    }
    public void setHyperParametersInfo(int batchSize, int epochs, double learningRate) {
        hyperParametersInfo.setBatchSize(batchSize);
        hyperParametersInfo.setEpochs(epochs);
        hyperParametersInfo.setLearningRate(learningRate);
    }
}

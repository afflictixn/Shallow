package home.gui;

import java.util.ArrayList;
import java.util.List;

import javafx.fxml.FXML;
import shallow.layers.Flatten;
import shallow.layers.ReLU;
import shallow.layers.Sigmoid;
import shallow.layers.configs.*;
import shallow.layers.weight_init.WeightInitEnum;
import shallow.losses.LossEnum;
import shallow.optimizers.Adam;
import shallow.optimizers.BaseOptimizer;
import shallow.optimizers.OptimizerEnum;
import shallow.optimizers.StochasticGradientDescent;

public class Connector {

    private static Connector instance;

    Connector(){
        instance = this;
    }
    public static Connector getInstance(){
        return instance;
    }


    DatasetEnum datasetEnum = DatasetEnum.CIFAR10;
    List<Config> configs = new ArrayList<>();
    HyperParametersInfo hyperParametersInfo = new HyperParametersInfo();
    LossEnum lossEnum = LossEnum.CategoricalCrossEntropyLoss;
    BaseOptimizer optimizer = new Adam();
    public void setDatasetEnum(DatasetEnum dataset) {
        datasetEnum = dataset;
    };
    public void setLossEnum(LossEnum lossEnum) {
        this.lossEnum = lossEnum;
    }
    public void setOptimizerAdam(double beta1, double beta2) {
        optimizer = new Adam(beta1, beta2);
    }
    public void setOptimizerSGD(double momentum){
        optimizer = new StochasticGradientDescent(momentum);
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
                              PaddingType paddingType, WeightInitEnum weight, WeightInitEnum bias) {
        configs.add(new Conv2dConfig().filters(filters)
                .kernelSize(kernelHeight, kernelWidth)
                .strides(stridesHeight, stridesWidth)
                .paddingType(paddingType).biasInitializer(bias).weightInitializer(weight));
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

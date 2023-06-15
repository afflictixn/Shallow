package home.gui;

public class HyperParametersInfo {
    int batchSize = 64;
    int epochs = 20;
    double learningRate = 0.05;
    double L2RegularizationLambda;

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpochs() {
        return epochs;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    public double getL2RegularizationLambda() {
        return L2RegularizationLambda;
    }
    public void setL2RegularizationLambda(double l2RegularizationLambda) {
        L2RegularizationLambda = l2RegularizationLambda;
    }
}

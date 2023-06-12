package home.gui;

public class HyperParametersInfo {
    int batchSize = 64;
    int epochs = 20;
    double learningRate = 0.05;

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
}

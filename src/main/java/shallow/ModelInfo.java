package shallow;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ModelInfo {
    boolean training = true;
    int currentEpoch;
    double currentLoss;
    int truePositive = 0, falsePositive = 0;
    int trueNegative = 0, falseNegative = 0;
    int totalPredictions = 0;
    double classificationThreshold = -1;
    public ModelInfo(double threshold) {
        classificationThreshold = threshold;
    }
    public ModelInfo(){}
    public void reset() {
        truePositive = falsePositive = trueNegative = falseNegative = 0;
        currentEpoch = totalPredictions = 0;
        currentLoss = 0;
    }
    public void setMetadata(int epoch, double loss){
        currentEpoch = epoch;
        currentLoss = loss;
    }
    public void evaluate(INDArray predictions, INDArray labels, boolean training) {
        this.training = training;
        INDArray nonPredictions = predictions.rsub(1.0);
        INDArray nonLabels = labels.rsub(1.0);
        int curTruePositive = predictions.mul(labels).sumNumber().intValue();
        int curFalsePositive = predictions.mul(nonLabels).sumNumber().intValue();
        int curFalseNegative = nonPredictions.mul(labels).sumNumber().intValue();
        int numEntries = (int) labels.shape()[0] * (int) labels.shape()[1];
        totalPredictions += (int) labels.shape()[0];
        trueNegative += numEntries - curTruePositive - curFalsePositive - curFalseNegative;
        truePositive += curTruePositive; falsePositive += curFalsePositive;
        falseNegative += curFalseNegative;
    }
    public void evaluateFromRaw(INDArray activation, INDArray labels) {
        INDArray mostProbablePredictions;
        if(classificationThreshold == -1){
            INDArray maxProba = activation.max(true, 1);
            mostProbablePredictions = activation.eq(maxProba).castTo(DataType.FLOAT);
        }
        else {
            mostProbablePredictions = activation.gt(classificationThreshold).castTo(activation.dataType());
        }
        evaluate(mostProbablePredictions, labels, true);
    }
    public double accuracy(){
        return (double) truePositive / totalPredictions;
    }
    public double precision(){
        return (double) truePositive / (truePositive + falsePositive);
    }
    public double recall() {
        return (double) truePositive / (truePositive + falseNegative);
    }
    public double f1Score(){
        double precision = precision();
        double recall = recall();
        return 2 * precision * recall / (precision + recall);
    }

    @Override
    public String toString() {
        if(training) {
            return String.format("Current loss: %f, Current accuracy: %f, current epoch: %d",
                    currentLoss,
                    accuracy(),
                    currentEpoch);
        }
        else {
            return String.format("Current accuracy: %f, Current precision: %f, Current F1Score: %f",
                    accuracy(),
                    precision(),
                    f1Score());
        }
    }
}

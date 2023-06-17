package shallow;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class ModelInfo {
    boolean training = true;

    public boolean isStopTraining() {
        return stopTraining;
    }

    public void setStopTraining(boolean stopTraining) {
        this.stopTraining = stopTraining;
    }

    boolean stopTraining = false;
    int currentEpoch, totalEpoch;
    double currentLoss;
    int truePositive = 0, falsePositive = 0;
    int trueNegative = 0, falseNegative = 0;
    int totalPredictions = 0;
    double classificationThreshold = -1;

    public ModelInfo() {
    }

    public ModelInfo(int totalEpoch, double threshold) {
        this.totalEpoch = totalEpoch;
        classificationThreshold = threshold;
    }

    public ModelInfo(int totalEpoch) {
        this.totalEpoch = totalEpoch;
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public int getTotalEpoch() {
        return totalEpoch;
    }

    public double getCurrentLoss() {
        return currentLoss;
    }

    public void reset() {
        truePositive = falsePositive = trueNegative = falseNegative = 0;
        currentEpoch = totalPredictions = 0;
        currentLoss = 0;
    }

    public void setMetadata(int epoch, double loss) {
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
        truePositive += curTruePositive;
        falsePositive += curFalsePositive;
        falseNegative += curFalseNegative;
    }

    public void evaluateFromRaw(INDArray activation, INDArray labels) {
        INDArray mostProbablePredictions;
        if (classificationThreshold == -1) {
            long[] maxProbaIndex = activation.argMax(1).toLongVector();
            mostProbablePredictions = Nd4j.zerosLike(activation);
            for(int i = 0; i < activation.shape()[0]; ++i) {
                mostProbablePredictions.get(point(i), point(maxProbaIndex[i])).addi(1.0);
            }
        } else {
            mostProbablePredictions = activation.gt(classificationThreshold).castTo(activation.dataType());
        }
        evaluate(mostProbablePredictions, labels, true);
    }

    public double accuracy() {
        if (totalPredictions == 0) {
            return 0;
        }
        return (double) truePositive / totalPredictions;
    }

    public double precision() {
        return (double) truePositive / (truePositive + falsePositive);
    }

    public double recall() {
        return (double) truePositive / (truePositive + falseNegative);
    }

    public double f1Score() {
        double precision = precision();
        double recall = recall();
        return 2 * precision * recall / (precision + recall);
    }

    @Override
    public String toString() {
        if (training) {
            return String.format("Current loss: %f, Current accuracy: %f, current epoch: %d",
                    currentLoss,
                    accuracy(),
                    currentEpoch);
        } else {
            return String.format("Current accuracy: %f, Current precision: %f, Current F1Score: %f",
                    accuracy(),
                    precision(),
                    f1Score());
        }
    }
}

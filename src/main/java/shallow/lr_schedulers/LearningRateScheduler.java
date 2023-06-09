package shallow.lr_schedulers;

public interface LearningRateScheduler {
    double getCurrentLearningRate(double initialRate, int epochNum);
}

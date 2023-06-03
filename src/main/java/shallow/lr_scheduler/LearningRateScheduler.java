package shallow.lr_scheduler;

public interface LearningRateScheduler {
    double getCurrentLearningRate(double initialRate, int epochNum);
}

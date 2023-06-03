package shallow.lr_scheduler;

public class ConstantScheduler implements LearningRateScheduler{
    @Override
    public double getCurrentLearningRate(double initialRate, int epochNum) {
        return initialRate;
    }
}

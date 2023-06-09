package shallow.lr_schedulers;

public class ConstantScheduler implements LearningRateScheduler{
    @Override
    public double getCurrentLearningRate(double initialRate, int epochNum) {
        return initialRate;
    }
}

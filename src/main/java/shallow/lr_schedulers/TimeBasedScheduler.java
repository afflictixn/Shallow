package shallow.lr_schedulers;

public class TimeBasedScheduler implements LearningRateScheduler {
    double stopFactor = 1e-5;
    double decay = 1.0;

    public TimeBasedScheduler() {

    }

    public TimeBasedScheduler(double decay) {
        this.decay = decay;
    }

    public TimeBasedScheduler(double decay, double stopFactor) {
        this.decay = decay;
        this.stopFactor = stopFactor;
    }

    @Override
    public double getCurrentLearningRate(double initialRate, int epochNum) {
        if (initialRate < stopFactor) {
            return initialRate;
        }
        return initialRate / (1 + decay * epochNum);
    }
}

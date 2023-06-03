package shallow.lr_scheduler;

public class IntervalBasedScheduler implements LearningRateScheduler {
    double stopFactor = 1e-5;
    double decay = 0.4;
    int timeInterval = 10;

    public IntervalBasedScheduler(double decay, int timeInterval, double stopFactor) {
        this.decay = decay;
        this.timeInterval = timeInterval;
        this.stopFactor = stopFactor;
    }

    public IntervalBasedScheduler(double decay) {
        this.decay = decay;
    }

    public IntervalBasedScheduler(int timeInterval) {
        this.timeInterval = timeInterval;
    }

    public void setStopFactor(double stopFactor) {
        this.stopFactor = stopFactor;
    }

    @Override
    public double getCurrentLearningRate(double initialRate, int epochNum) {
        if(initialRate < stopFactor) {
            return initialRate;
        }
        return initialRate / (1 + decay * Math.floor(epochNum / (double) timeInterval));
    }
}

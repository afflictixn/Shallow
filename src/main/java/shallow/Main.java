package shallow;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.*;
import shallow.layers.configs.Conv2dConfig;
import shallow.layers.configs.LinearLayerConfig;
import shallow.layers.configs.MaxPool2dConfig;
import shallow.layers.configs.PaddingType;
import shallow.layers.weight_init.WeightInitEnum;
import shallow.losses.CategoricalCrossEntropyLoss;
import shallow.lr_schedulers.IntervalBasedScheduler;
import shallow.lr_schedulers.LearningRateScheduler;
import shallow.optimizers.Adam;


public class Main {
    public static void begin(){
        int batchSize = 2;
        int height = 6;
        int width = 6;
        int channels = 1;
        INDArray input = Nd4j.create(batchSize, height, width, channels);

        Model mod = new Model();
        mod.addLayer(new Conv2d(new Conv2dConfig().kernelSize(3, 3)
                .strides(1, 1).paddingType(PaddingType.SAME).weightInitializer(WeightInitEnum.HeNormal).filters(5)));
        mod.addLayer(new ReLU());
        mod.addLayer(new MaxPool2d(new MaxPool2dConfig().kernelSize(2, 2).strides(2, 2)));
        mod.addLayer(new Flatten());
        mod.addLayer(new Linear(new LinearLayerConfig()
                .units(10)
                .weightInitializer(WeightInitEnum.HeNormal)));
        mod.addLayer(new ReLU());

        mod.setLoss(new CategoricalCrossEntropyLoss());
        mod.setOptimizer(new Adam());
        LearningRateScheduler scheduler = new IntervalBasedScheduler(0.2, 100, 0.005);
        mod.setScheduler(scheduler);
    }
    public static void main(String[] args) {
        begin();
//        mod.fit(resh, labels, 0.09, 32, 10);
    }
}
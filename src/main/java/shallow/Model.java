package shallow;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.BaseLayer;
import shallow.layers.ShapeChangingLayer;
import shallow.layers.WeightedLayer;
import shallow.losses.BaseLoss;
import shallow.losses.BinaryCrossEntropyLoss;
import shallow.losses.CategoricalCrossEntropyLoss;
import shallow.lr_schedulers.ConstantScheduler;
import shallow.lr_schedulers.LearningRateScheduler;
import shallow.optimizers.BaseOptimizer;
import shallow.utils.Utils;

import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

class MiniBatch {
    INDArray X; // features data
    INDArray Y; // labels data

    MiniBatch(INDArray X, INDArray Y) {
        this.X = X;
        this.Y = Y;
    }
}

public class Model {
    List<BaseLayer> layers;
    List<WeightedLayer> trainableLayers;
    List<ShapeChangingLayer> shapeChangingLayers;
    BaseLoss loss;
    BaseOptimizer optimizer;
    LearningRateScheduler scheduler = new ConstantScheduler();
    double L2RegularizationLambda; // hyperparameter for L2 Regularization of a model to reduce overfitting
    public boolean isNCHWOrder = false; // specifies whether the input is given in [batchSize, Channels, Height, Width] format
    final ModelInfo info;
    public Model() {
        layers = new ArrayList<>();
        trainableLayers = new ArrayList<>();
        shapeChangingLayers = new ArrayList<>();
        this.info = new ModelInfo();
    }
    public Model(ModelInfo info) {
        layers = new ArrayList<>();
        trainableLayers = new ArrayList<>();
        shapeChangingLayers = new ArrayList<>();
        this.info = info;
    }

    public Model addLayer(BaseLayer layer) {
        layers.add(layer);
        if (layer instanceof ShapeChangingLayer shapeChangingLayer) {
            shapeChangingLayers.add(shapeChangingLayer);
        }
        if (layer instanceof WeightedLayer weightedLayer) {
            trainableLayers.add(weightedLayer);
        }
        return this;
    }

    public Model setLoss(BaseLoss loss) {
        this.loss = loss;
        return this;
    }

    public Model setOptimizer(BaseOptimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    public Model setScheduler(LearningRateScheduler scheduler) {
        this.scheduler = scheduler;
        return this;
    }

    public Model setL2Regularization(double lambda) {
        if(lambda < 0.0){
            throw new IllegalArgumentException("Lambda for L2 Regularization has to be non negative");
        }
        L2RegularizationLambda = lambda;
        return this;
    }

    // sequential forward pass of a model
    public INDArray forwardPass(INDArray X) {
        for (BaseLayer layer : layers) {
            X = layer.forward(X);
        }
        return X;
    }

    public double computeLoss(INDArray X, INDArray Y) {
        double totalLoss = loss.forward(X, Y).getDouble();
        // add L2 Regularization penalty to the loss
        if(L2RegularizationLambda != 0.0) {
            for (WeightedLayer weightedLayer : trainableLayers) {
                totalLoss -= Utils.get().square(weightedLayer.getWeightValues().dup())
                        .muli(L2RegularizationLambda / 2).sumNumber().doubleValue();
            }
        }
        return totalLoss;
    }

    public void backwardPass() {
        INDArray D = loss.backward();
        for (int i = layers.size() - 1; i >= 0; --i) {
            D = layers.get(i).backward(D);
        }
    }
    public INDArray getProbas(INDArray X) {
        INDArray res = forwardPass(X);
        if(loss.getClass().equals(CategoricalCrossEntropyLoss.class)) {
            res = Utils.softmax(res);
        }
        return res;
    }
    public INDArray predict(INDArray X) {
        INDArray res = getProbas(X);
        INDArray maxProba = res.max(true, 1);
        return res.eq(maxProba).castTo(DataType.FLOAT);
    }
    public void evaluateTestSet(INDArray testFeatures, INDArray testLabels) {
        testFeatures = (isNCHWOrder) ? testFeatures.permute(0, 2, 3, 1) : testFeatures;
        INDArray forwardPass = forwardPass(testFeatures);
        double lossValue = -computeLoss(forwardPass, testLabels) / info.totalPredictions;
        info.reset();
        info.evaluateFromRaw(loss.getActivation(), testLabels);

        info.setMetadata(0, lossValue);
    }
    void prepareModel(long... inputShape){
        if (!shapeChangingLayers.isEmpty()) {
            shapeChangingLayers.get(0).init(inputShape);
            for (int i = 1; i < shapeChangingLayers.size(); ++i) {
                shapeChangingLayers.get(i).init(shapeChangingLayers.get(i - 1).getOutputShape());
            }
        }
        optimizer.addRegularizationL2(L2RegularizationLambda);
        optimizer.init(trainableLayers);
    }
    // trains model on one mini batch
    // returns loss of a current mini batch iteration
    double oneStep(INDArray X, INDArray Y, double currentLearningRate, int currentEpoch) {
        INDArray result = forwardPass(X);
        double stepLoss = computeLoss(result, Y);
        backwardPass();
        optimizer.updateWeights(currentLearningRate, currentEpoch);

        info.evaluateFromRaw(loss.getActivation(), Y);
        return stepLoss;
    }

    // 0-th dimension of X and Y has a size of a number of samples
    public void fit(INDArray X, INDArray Y, double learningRate, int batchSize, int numEpochs) {
        X = (isNCHWOrder) ? X.permute(0, 2, 3, 1) : X;
        prepareModel(X.shape());
        long numSamples = X.shape()[0];
        List<MiniBatch> miniBatches = (X.shape().length == 2) ?
                randomMiniBatches(X, Y, batchSize) : getMiniBatches(X, Y, batchSize);

        for (int currentEpoch = 1; currentEpoch <= numEpochs; ++currentEpoch) {
            info.reset();
            double totalLoss = 0.0;
            double currentLearningRate = scheduler.getCurrentLearningRate(learningRate, currentEpoch);
            for (MiniBatch miniBatch : miniBatches) {
                totalLoss += oneStep(miniBatch.X, miniBatch.Y, currentLearningRate, currentEpoch);
                System.out.println(info);
            }
            totalLoss /= -numSamples;
            info.setMetadata(currentEpoch, totalLoss);
            System.out.println(info);
            synchronized (info) {
                info.notify();
            }
            if(info.stopTraining){
                return;
            }
        }
    }

    public void fit(DataSetIterator iterator, double learningRate, int numEpochs) {
        DataSet dataSet = iterator.next();
        INDArray features = (isNCHWOrder) ? dataSet.getFeatures().permute(0, 2, 3, 1) : dataSet.getFeatures();
        prepareModel(features.shape());
        for (int currentEpoch = 1; currentEpoch <= numEpochs; ++currentEpoch) {
            info.reset();
            double totalLoss = 0.0;
            double currentLearningRate = scheduler.getCurrentLearningRate(learningRate, currentEpoch);
            while(true) {
                totalLoss += oneStep(features, dataSet.getLabels(), currentLearningRate, currentEpoch);
                if(!iterator.hasNext()){
                    break;
                }
                dataSet = iterator.next();
                features = (isNCHWOrder) ? dataSet.getFeatures().permute(0, 2, 3, 1) : dataSet.getFeatures();
            }
            totalLoss /= -info.totalPredictions;
            info.setMetadata(currentEpoch, totalLoss);
            System.out.println(info);
            synchronized (info) {
                info.notify();
            }
            iterator.reset();
            if(info.isStopTraining()){
                return;
            }
        }
    }

    private static List<MiniBatch> randomMiniBatches(INDArray X, INDArray Y, int batchSize) {
        int numSamples = (int) X.shape()[0];
        int[] permutation = Utils.randomPermutation(numSamples);
        INDArray X_shuffle = X.getRows(permutation);
        INDArray Y_shuffle = Y.getRows(permutation);
        return getMiniBatches(X_shuffle, Y_shuffle, batchSize);
    }

    private static List<MiniBatch> getMiniBatches(INDArray X, INDArray Y, int batchSize) {
        int numSamples = (int) X.shape()[0];
        int numCompleteBatches = numSamples / batchSize;
        List<MiniBatch> miniBatches = new LinkedList<>();
        for (int i = 0; i < numCompleteBatches; ++i) {
            INDArray X_mini = X.get(interval(i * batchSize, (i + 1) * batchSize));
            INDArray Y_mini = Y.get(interval(i * batchSize, (i + 1) * batchSize));
            miniBatches.add(new MiniBatch(X_mini, Y_mini));
        }
        int wholeSamples = numCompleteBatches * batchSize;
        if (numSamples % batchSize != 0) {
            INDArray X_mini = X.get(interval(wholeSamples, numSamples));
            INDArray Y_mini = Y.get(interval(wholeSamples, numSamples));
            miniBatches.add(new MiniBatch(X_mini, Y_mini));
        }
        return miniBatches;
    }
}

package shallow;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.layers.BaseLayer;
import shallow.layers.ShapeChangingLayer;
import shallow.layers.WeightedLayer;
import shallow.losses.BaseLoss;
import shallow.losses.BinaryCrossEntropyLoss;
import shallow.losses.CategoricalCrossEntropyLoss;
import shallow.lr_schedulers.LearningRateScheduler;
import shallow.optimizers.BaseOptimizer;
import shallow.utils.Utils;

import java.util.*;

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
    LearningRateScheduler scheduler;
    private int lastLayerSize;

    public Model() {
        layers = new ArrayList<>();
        trainableLayers = new ArrayList<>();
        shapeChangingLayers = new ArrayList<>();
    }

    public void addLayer(BaseLayer layer) {
        if (!validLayerCheck(layer)) {
            throw new IllegalArgumentException();
        }
        layers.add(layer);
        if (layer instanceof ShapeChangingLayer shapeChangingLayer) {
            shapeChangingLayers.add(shapeChangingLayer);
        }
        if (layer instanceof WeightedLayer weightedLayer) {
            trainableLayers.add(weightedLayer);
        }
    }

    public void setLoss(BaseLoss loss) {
        this.loss = loss;
    }

    public void setOptimizer(BaseOptimizer optimizer) {
        this.optimizer = optimizer;
    }

    public void setScheduler(LearningRateScheduler scheduler) {
        this.scheduler = scheduler;
    }

    // sequential forward pass of a model
    public INDArray forwardPass(INDArray X) {
        for (BaseLayer layer : layers) {
            X = layer.forward(X);
        }
        return X;
    }

    public double computeLoss(INDArray X, INDArray Y) {
        return loss.forward(X, Y).getDouble();
    }

    public void backwardPass() {
        INDArray D = loss.backward();
        for (int i = layers.size() - 1; i >= 0; --i) {
            D = layers.get(i).backward(D);
        }
    }

    public INDArray predict(INDArray X) {
        if (loss.getClass().equals(BinaryCrossEntropyLoss.class) || loss.getClass().equals(CategoricalCrossEntropyLoss.class)) {
            // transforms probabilities given by forward pass of network into predictions 1 or 0
            INDArray probs = forwardPass(X);
            INDArray ans = probs.gt(0.5).castTo(DataType.FLOAT);
            return ans;
        }
        return null;
    }

    // 0-th dimension of X and Y is a number of samples
    public void fit(INDArray X, INDArray Y, double learningRate, int batchSize, int numEpochs) {
        if (!shapeChangingLayers.isEmpty()) {
            shapeChangingLayers.get(0).init(X.shape());
            for (int i = 1; i < shapeChangingLayers.size(); ++i) {
                shapeChangingLayers.get(i).init(shapeChangingLayers.get(i - 1).getOutputShape());
            }
        }
        optimizer.init(trainableLayers);
        INDArray result = null;
        long numSamples = X.shape()[0];
        List<MiniBatch> miniBatches = (X.shape().length == 2) ?
                randomMiniBatches(X, Y, batchSize) : getMiniBatches(X, Y, batchSize);
        for (int i = 1; i <= numEpochs; ++i) {
            double totalCost = 0.0;
            double currentLearningRate = scheduler.getCurrentLearningRate(learningRate, i);
            for (MiniBatch miniBatch : miniBatches) {
                result = forwardPass(miniBatch.X);
                totalCost += computeLoss(result, miniBatch.Y);
                backwardPass();
                optimizer.updateWeights(currentLearningRate, i);
            }
            totalCost /= -numSamples;

            // ToDo make class for display of metrics and data
            if (i % 25 == 1 || i == numEpochs) {
                System.out.println("Current loss: " + totalCost);
                System.out.println("Current learning rate: " + currentLearningRate);
            }
        }
    }

    private boolean validLayerCheck(BaseLayer layer) {
        boolean ok = true;
//        if (layer instanceof WeightedLayer weighted) {
//            ok = (last_layer_size == 0) || last_layer_size == weighted.getInputSize();
//            last_layer_size = weighted.getOutputSize();
//        }
        return ok;
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
        long[] X_shape = X.shape().clone();
        X_shape[0] = batchSize;
        long[] Y_shape = Y.shape().clone();
        Y_shape[0] = batchSize;
        int cnt = 0;
        for (int i = 0; i < numCompleteBatches; ++i) {
            INDArray X_mini = Nd4j.create(X_shape);
            INDArray Y_mini = Nd4j.create(Y_shape);
            for (int j = 0; j < batchSize; ++j) {
                X_mini.putSlice(j, X.slice(cnt));
                if (Y_mini.isVector()) {
                    Y_mini.put(j, Y.slice(cnt++));
                } else {
                    Y_mini.putSlice(j, Y.slice(cnt++));
                }
            }
            miniBatches.add(new MiniBatch(X_mini, Y_mini));
        }
        int left = numSamples - numCompleteBatches * batchSize;
        X_shape[0] = left;
        Y_shape[0] = left;
        if (numSamples % batchSize != 0) {
            INDArray X_mini = Nd4j.create(X_shape);
            INDArray Y_mini = Nd4j.create(Y_shape);
            for (int j = 0; j < left; ++j) {
                X_mini.putSlice(j, X.slice(cnt));
                if (Y_mini.isVector()) {
                    Y_mini.put(j, Y.slice(cnt++));
                } else {
                    Y_mini.putSlice(j, Y.slice(cnt++));
                }
            }
        }
        return miniBatches;
    }
}

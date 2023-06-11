package home.gui;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public enum DatasetEnum {
    UNKNOWN,
    MNIST,
    CIFAR10;
    static class Cifar10Parameters {
        public static final int HEIGHT = 32;
        public static final int WIDTH = 32;
    }
    public DataSetIterator getTrainDataSetIterator(int batchSize, int seed) throws IOException {
        switch (this){
            case MNIST -> {
                return new MnistDataSetIterator(batchSize, true, seed);
            }
            case CIFAR10 -> {
                return new Cifar10DataSetIterator(batchSize,
                        new int[]{DatasetEnum.Cifar10Parameters.HEIGHT, DatasetEnum.Cifar10Parameters.WIDTH},
                        DataSetType.TRAIN, null, seed);
            }
        }
        throw new RuntimeException();
    }
    public DataSetIterator getTestDataSetIterator(int batchSize, int seed) throws IOException {
        switch (this){
            case MNIST -> {
                return new MnistDataSetIterator(batchSize, false, seed);
            }
            case CIFAR10 -> {
                return new Cifar10DataSetIterator(batchSize,
                        new int[]{DatasetEnum.Cifar10Parameters.HEIGHT, DatasetEnum.Cifar10Parameters.WIDTH},
                        DataSetType.TEST, null, seed);
            }
        }
        throw new RuntimeException();
    }
}

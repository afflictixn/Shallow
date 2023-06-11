/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.ConfusionMatrix;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import shallow.Model;
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

/**
 * train model by cifar
 * identification unknown file
 *
 * @author wangfeng
 * @since June 7,2017
 */

//@Slf4j
@SuppressWarnings("FieldCanBeLocal")
public class CIFARClassifier {
    protected static final Logger log = LoggerFactory.getLogger(CIFARClassifier.class);

    private static int height = 32;
    private static int width = 32;
    private static int channels = 3;
    private static int numLabels = CifarLoader.NUM_LABELS;
    private static int batchSize = 96;
    private static long seed = 123L;
    private static int epochs = 4;

    public static void main(String[] args) throws Exception {
        CIFARClassifier cf = new CIFARClassifier();
        Cifar10DataSetIterator cifar = new Cifar10DataSetIterator(batchSize, new int[]{height, width}, DataSetType.TRAIN, null, seed);
        Cifar10DataSetIterator cifarEval = new Cifar10DataSetIterator(batchSize, new int[]{height, width}, DataSetType.TEST, null, seed);
        DataSet dat = cifar.next();
        INDArray resh = dat.getFeatures().permute(0, 2, 3, 1);
        INDArray labels = dat.getLabels();

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

        mod.fit(resh, labels, 0.09, 32, 10);
//        //train model and eval model
//        MultiLayerNetwork model = cf.getModel();
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
//        uiServer.attach(statsStorage);
//        model.setListeners(new StatsListener( statsStorage), new ScoreIterationListener(50), new EvaluativeListener(cifarEval, 1, InvocationType.EPOCH_END));
//
//        model.fit(cifar, epochs);
//
//        log.info("Saving model...");
//        model.save(new File(System.getProperty("java.io.tmpdir"), "cifarmodel.dl4j.zip"), true);

        System.exit(0);
    }


    public MultiLayerNetwork getModel()  {
        log.info("Building simple convolutional network...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new AdaDelta())
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nIn(channels).nOut(32).build())
            .layer(new BatchNormalization())
            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

            .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nOut(16).build())
            .layer(new BatchNormalization())
            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nOut(64).build())
            .layer(new BatchNormalization())
            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

            .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nOut(32).build())
            .layer(new BatchNormalization())
            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nOut(128).build())
            .layer(new BatchNormalization())
            .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nOut(64).build())
            .layer(new BatchNormalization())
            .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                .nOut(numLabels).build())
            .layer(new BatchNormalization())

            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())

            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .dropOut(0.8)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

}


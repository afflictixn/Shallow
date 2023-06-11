import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import shallow.Model;
import shallow.losses.CategoricalCrossEntropyLoss;
import shallow.lr_schedulers.IntervalBasedScheduler;
import shallow.lr_schedulers.LearningRateScheduler;
import shallow.optimizers.StochasticGradientDescent;

import java.io.File;


@SuppressWarnings("DuplicatedCode")
public class MoonClassifier {
    private static int height = 32;
    private static int width = 32;
    private static int channels = 3;
    private static int numLabels = CifarLoader.NUM_LABELS;
    private static int batchSize = 96;
    private static long seed = 123L;
    private static int epochs = 4;
    public static boolean visualize = true;
    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {
        double learningRate = 0.001;
        int batchSize = 2000;
        int nEpochs = 700;
        int numInputs = 2;
        int numOutputs = 2;
        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();

//        Cifar10DataSetIterator cifar = new Cifar10DataSetIterator(batchSize, new int[]{height, width}, DataSetType.TRAIN, null, seed);
//        Cifar10DataSetIterator cifarEval = new Cifar10DataSetIterator(batchSize, new int[]{height, width}, DataSetType.TEST, null, seed);

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dataLocalPath,"moon_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(dataLocalPath,"moon_data_eval.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

        //Build model
        Model mod = new Model();
//        mod.addLayer(new Linear(new LinearLayerConfig()
//                .inputSize(numInputs)
//                .outputSize(50)
//                .weightInitializer(WeightInitEnum.HeNormal)));
//        mod.addLayer(new ReLU());
//        mod.addLayer(new Linear(new LinearLayerConfig()
//                .inputSize(50)
//                .outputSize(numOutputs)
//                .weightInitializer(WeightInitEnum.XavierNormal)));
        mod.setLoss(new CategoricalCrossEntropyLoss());
        mod.setOptimizer(new StochasticGradientDescent());
        LearningRateScheduler scheduler = new IntervalBasedScheduler(0.2, 100, 0.005);
        mod.setScheduler(scheduler);
        DataSet dataset = trainIter.next();
        INDArray labels = dataset.getLabels();
//        System.out.println(Arrays.toString(labels.shape()));
//        System.out.println(Arrays.toString(dataset.getFeatures().shape()));
        mod.fit(dataset.getFeatures(), labels, 0.09, 32, nEpochs);

        generateVisuals(mod, trainIter, testIter);
    }

    public static void generateVisuals(Model model, DataSetIterator trainIter, DataSetIterator testIter) throws Exception {
        if (visualize) {
            double xMin = -1.5;
            double xMax = 2.5;
            double yMin = -1;
            double yMax = 1.5;

            //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
            int nPointsPerAxis = 100;

            //Generate x,y points that span the whole range of features
            INDArray allXYPoints = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis);
            //Get train data and plot with predictions
//            PlotUtil.plotTrainingData(model, trainIter, allXYPoints, nPointsPerAxis);
//            TimeUnit.SECONDS.sleep(1);
            //Get test data, run the test data through the network to generate predictions, and plot those predictions:
            PlotUtil.plotTestData(model, testIter, allXYPoints, nPointsPerAxis);
        }
    }

}

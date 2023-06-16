package home.gui;


import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import home.gui.EvaluationUtils.*;
import shallow.Main;
import shallow.Model;
import shallow.ModelInfo;
import shallow.layers.configs.Config;


import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URL;
import java.util.*;

import static home.gui.EvaluationUtils.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class TestSetController implements Initializable {

    public final static int width = 100;
    public final static int height = 100;
    INDArray currentFeature;
    DataSetIterator iteratorShow = MainController.getConnector()
            .datasetEnum.getTestDataSetIterator(1, 123);
    @FXML
    private Button evaluateButton;

    @FXML
    private Label lossResult; // primarily is "-"

    @FXML
    private Label accuracyResult; // primarily is "-"

    @FXML
    private Canvas canvas;

    @FXML
    private Button nextButton;

    @FXML
    private Button predictButton;

    @FXML
    private Label prediction1; // this is number 1 with prediction "y"

    @FXML
    private Label prediction2; // this is number 2 with prediciton "x"

    @FXML
    private Label superAnswer; // big label, on which we will be showing the result of our calculations

    @FXML
    private Button Return;

    public TestSetController() throws IOException {
    }

    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    }


    public void nextButton() {
        if (!iteratorShow.hasNext()) {
            iteratorShow.reset();
        }
        currentFeature = iteratorShow.next().getFeatures();
        if (MainController.getConnector().datasetEnum.equals(DatasetEnum.MNIST)) {
            INDArray transform = currentFeature.reshape(1, DatasetEnum.MnistParameters.HEIGHT, DatasetEnum.MnistParameters.WIDTH, 1);
            transform = Nd4j.image().imageResize(transform, Nd4j.create(new double[]{canvas.getHeight(), canvas.getWidth()})
                    .castTo(DataType.INT8), false, true, ImageResizeMethod.ResizeBilinear);
            transform = transform.reshape((int) canvas.getHeight(), (int) canvas.getWidth());
            double[][] grayMatrix = transform.toDoubleMatrix();
            showPicture(canvas, grayMatrix);
        } else if (MainController.getConnector().datasetEnum.equals(DatasetEnum.CIFAR10)) {
            INDArray transform = currentFeature.permute(0, 2, 3, 1);
            transform = Nd4j.image().imageResize(transform, Nd4j.create(new double[]{canvas.getHeight(), canvas.getWidth()})
                    .castTo(DataType.INT8), false, true, ImageResizeMethod.ResizeBilinear);
            double[][] redMatrix = transform.get(point(0), all(), all(), point(0)).toDoubleMatrix();
            double[][] greenMatrix = transform.get(point(0), all(), all(), point(1)).toDoubleMatrix();
            double[][] blueMatrix = transform.get(point(0), all(), all(), point(2)).toDoubleMatrix();
            showColorPicture(canvas, redMatrix, greenMatrix, blueMatrix);
        }
    }

    public void predictButton() {
        INDArray prediction = MainController.neuralNetworkModel.getProbas(currentFeature);
        double[] probas = prediction.toDoubleVector();
        int[] sortedIndexes = getTop3Probas(probas);

        setProbabilityLabel(superAnswer, sortedIndexes[0], probas[sortedIndexes[0]]);
        setProbabilityLabel(prediction1, sortedIndexes[1], probas[sortedIndexes[1]]);
        setProbabilityLabel(prediction2, sortedIndexes[2], probas[sortedIndexes[2]]);
    }

    public void evaluateButton() throws IOException {
        DataSetIterator iter;
        if(MainController.getConnector().datasetEnum.equals(DatasetEnum.CIFAR10)){
            iter = MainController.getConnector().datasetEnum.getTestDataSetIterator(DatasetEnum.Cifar10Parameters.TESTING_EXAMPLES, 123);
        }
        else {
            iter = MainController.getConnector().datasetEnum.getTestDataSetIterator(10000, 123);
        }
        DataSet test = iter.next();
        INDArray features = (MainController.getConnector().datasetEnum.equals(DatasetEnum.CIFAR10)) ?
                test.getFeatures().div(255) : test.getFeatures();
        MainController.neuralNetworkModel.evaluateTestSet(features, test.getLabels());
        lossResult.setText(String.valueOf(MainController.modelInfo.getCurrentLoss()));
        accuracyResult.setText(String.valueOf(MainController.modelInfo.accuracy()));
    }

    public void refreshLabels() {
        // TODO the same as in EvaluateController
    }

    @FXML
    private AnchorPane innerAnchorPane;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        //innerAnchorPane.setVisible(false); TODO should be uncommented
    }
}

package home.gui;

import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.application.Application;
import javafx.fxml.Initializable;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.eclipse.collections.impl.block.factory.Comparators;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import shallow.Model;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URL;
import java.util.Arrays;
import java.util.Comparator;
import java.util.ResourceBundle;
import java.util.stream.IntStream;

public class EvaluateController implements Initializable {

    @FXML
    private Canvas canvas;

    @FXML
    private Label superAnswer;

    @FXML
    private Button startButton;

    @FXML
    private Button finishButton;

    @FXML
    private Button clearButton;

    private GraphicsContext graphicsContext;
    private boolean drawingEnabled = false;

    public double[][] matrix;


    private void startDrawing() {
        drawingEnabled = true;
        System.out.println("start");
    }

    private void finishDrawing() throws IOException {
        drawingEnabled = false;
        processCanvasData();
        System.out.println("finish");
    }

    private void clearCanvas() {
        graphicsContext.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        System.out.println("clear");
    }

    private void processCanvasData() throws IOException {
        int width = (int) canvas.getWidth();
        int height = (int) canvas.getHeight();

        matrix = new double[height][width];

        WritableImage snapshot = canvas.snapshot(null, null);
        PixelReader pixelReader = snapshot.getPixelReader();

        System.out.println("EVALUATE:");
        DataSetIterator iter = DatasetEnum.MNIST.getTestDataSetIterator(10000, 123);
        Model model = MainController.neuralNetworkModel;
        DataSet test = iter.next();
        model.evaluateTestSet(test.getFeatures(), test.getLabels());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = pixelReader.getColor(x, y);
                double grayscaleValue = color.getRed() * 0.299 + color.getGreen() * 0.587 + color.getBlue() * 0.114;
//                matrix[y][x] = color.equals(Color.BLACK) ? 1 : 0;
                matrix[y][x] = 1.0 - grayscaleValue;
            }
        }
        System.out.println("evaluate: ");


        INDArray image = Nd4j.create(matrix);
        image = image.reshape(1, image.shape()[0], image.shape()[1], 1);
        image = Nd4j.image().imageResize(image, Nd4j.create(new double[]{28, 28}).castTo(DataType.INT8),
                false, true, ImageResizeMethod.ResizeBilinear);
        image = image.reshape(28, 28);
        double[][] res = image.toDoubleMatrix();
        image = image.reshape(1, image.shape()[0] * image.shape()[1]);

        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                System.out.print(BigDecimal.valueOf(res[i][j]).setScale(1, RoundingMode.HALF_DOWN) + " ");
            }
            System.out.println();
        }

        INDArray prediction = model.getProbas(image);
        double[] probas = prediction.toDoubleVector();
        // Sort indexes array in ascending order based on probas values
        int[] sortedIndexes = IntStream.range(0, MainController.getConnector().datasetEnum.getNumberOfClasses())
                .boxed().sorted(Comparator.comparing(i -> probas[i])).mapToInt(element -> element).toArray();

        for (int i = probas.length - 1; i >= probas.length - 3; --i) {
            System.out.println("Probability of " + sortedIndexes[i] + ": " + probas[sortedIndexes[i]]);
        }
        double label = prediction.argMax(0, 1).getDouble();
        System.out.println("Label: " + label);
        superAnswer.setText(Double.toString(label));
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        graphicsContext = canvas.getGraphicsContext2D();
        graphicsContext.setStroke(Color.BLACK);
        graphicsContext.setLineWidth(6);
        // Here I can choose the width of the line

        canvas.setOnMousePressed(event -> {
            if (drawingEnabled) {
                graphicsContext.beginPath();
                graphicsContext.moveTo(event.getX(), event.getY());
                graphicsContext.setStroke(Color.BLACK);
            }
        });

        canvas.setOnMouseDragged(event -> {
            if (drawingEnabled) {
                graphicsContext.lineTo(event.getX(), event.getY());
                graphicsContext.stroke();
            }
        });

        startButton.setOnAction(event -> startDrawing());
        finishButton.setOnAction(event -> {
            try {
                finishDrawing();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        clearButton.setOnAction(event -> clearCanvas());
    }
}

package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import shallow.Model;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

import static home.gui.EvaluationUtils.getTop3Probas;
import static home.gui.EvaluationUtils.setProbabilityLabel;

public class CanvasController implements Initializable {


    @FXML
    private Canvas canvas;


    @FXML
    private Button startButton;

    @FXML
    private Button finishButton;

    @FXML
    private Button clearButton;

    @FXML
    private Label superAnswer; // our big label, on which we are displaying predicted answer

    @FXML
    private Label prediction1; // this is number 1 with probability "x"

    @FXML
    private Label prediction2; // this is number 2 with probability "y"

//    public void refreshLabels() {
//
//    }

    @FXML
    private AnchorPane innerAnchorPane;

    @FXML
    private Button Return;

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("EvaluateMiddleClass.fxml");
    }

    private GraphicsContext graphicsContext;
    private boolean drawingEnabled = false;

    public double[][] matrix;




    private void startDrawing() {
        drawingEnabled = true;
        clearCanvas();
        innerAnchorPane.setVisible(false); // hiding additional information
        System.out.println("start");
    }

    private void finishDrawing() throws IOException {
        drawingEnabled = false;
        processCanvasData();
        innerAnchorPane.setVisible(true); // showing additional information
        System.out.println("finish");
    }

    public void clearCanvas() {
        graphicsContext.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        innerAnchorPane.setVisible(false);
    }

    private void processCanvasData() throws IOException {
        int width = (int) canvas.getWidth();
        int height = (int) canvas.getHeight();
        matrix = new double[height][width];
        WritableImage snapshot = canvas.snapshot(null, null);
        PixelReader pixelReader = snapshot.getPixelReader();

        System.out.println("EVALUATE:");
        Model model = MainController.neuralNetworkModel;


        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = pixelReader.getColor(x, y);
                double grayscaleValue = color.getRed() * 0.299 + color.getGreen() * 0.587 + color.getBlue() * 0.114;
//                matrix[y][x] = color.equals(Color.BLACK) ? 1 : 0;
                matrix[y][x] = 1.0 - grayscaleValue;
            }
        }


        INDArray image = Nd4j.create(matrix);
        image = image.reshape(1, image.shape()[0], image.shape()[1], 1);
        image = Nd4j.image().imageResize(image, Nd4j.create(new double[]{28, 28}).castTo(DataType.INT8),
                false, true, ImageResizeMethod.ResizeBilinear);
        image = image.reshape(28, 28);
        image = image.reshape(1, image.shape()[0] * image.shape()[1]);

        INDArray prediction = model.getProbas(image);
        double[] probas = prediction.toDoubleVector();
        int[] sortedIndexes = getTop3Probas(probas);

        setProbabilityLabel(superAnswer, sortedIndexes[0], probas[sortedIndexes[0]]);
        setProbabilityLabel(prediction1, sortedIndexes[1], probas[sortedIndexes[1]]);
        setProbabilityLabel(prediction2, sortedIndexes[2], probas[sortedIndexes[2]]);

        innerAnchorPane.setVisible(true);
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

        clearCanvas();
        drawingEnabled = false;

        innerAnchorPane.setVisible(false);
    }
}

package home.gui;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.paint.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import shallow.Model;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Comparator;
import java.util.stream.IntStream;

public class EvaluationUtils {
    public static int[] getTop3Probas(double[] probas) {
        // Sort indexes array in ascending order based on probas values
        return IntStream.range(0, MainController.getConnector().datasetEnum.getNumberOfClasses())
                .boxed().sorted(Comparator.comparing(i -> -probas[i])).mapToInt(element -> element).limit(3).toArray();
    }

    public static void setProbabilityLabel(Label label, int index, double proba){
        label.setText("This is " + index + " with probability " + BigDecimal.valueOf(proba).setScale(3, RoundingMode.HALF_DOWN));
    }

    public static void showPicture(Canvas canvas, double[][] matrix) { // this function shows the picture on the matrix
        int height = (int) canvas.getHeight();
        int width = (int) canvas.getWidth();
        GraphicsContext gc = canvas.getGraphicsContext2D();

        for(int i = 0; i<height; i++){
            for(int j = 0; j<width; j++){
                double value = matrix[i][j];
                int grayValue = (int) (value * 255);

                Color color = Color.rgb(grayValue, grayValue, grayValue);
                gc.setFill(color);
                gc.fillRect(j, i, 1, 1);
            }
        }
    }
    public static void showColorPicture(Canvas canvas, double[][] featureRed, double[][] featureGreen, double[][] featureBlue) {
        int height = (int) canvas.getHeight();
        int width = (int) canvas.getWidth();
        GraphicsContext gc = canvas.getGraphicsContext2D();

        for(int i = 0; i<height; i++){
            for(int j = 0; j<width; j++){
                int red = (int) featureRed[i][j];
                int green = (int) featureGreen[i][j];
                int blue = (int) featureBlue[i][j];

                Color color = Color.rgb(red, green, blue);
                gc.setFill(color);
                gc.fillRect(j, i, 1, 1);
            }
        }
    }
}

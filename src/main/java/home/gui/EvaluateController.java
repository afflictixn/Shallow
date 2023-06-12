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
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URL;
import java.util.Arrays;
import java.util.ResourceBundle;

public class EvaluateController implements Initializable {

    @FXML
    private Canvas canvas;

    @FXML
    private Button startButton;

    @FXML
    private Button finishButton;

    @FXML
    private Button clearButton;

    private GraphicsContext graphicsContext;
    private boolean drawingEnabled = false;

    public double matrix[][];


    private void startDrawing() {
        drawingEnabled = true;
        System.out.println("start");
    }

    private void finishDrawing() {
        drawingEnabled = false;
        processCanvasData();
        System.out.println("finish");
    }

    private void clearCanvas() {
        graphicsContext.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        System.out.println("clear");
    }

    private void processCanvasData() {
        int width = (int) canvas.getWidth();
        int height = (int) canvas.getHeight();

        matrix = new double[height][width]; // or maybe I should change them ?


        WritableImage snapshot = canvas.snapshot(null, null);
        PixelReader pixelReader = snapshot.getPixelReader();


        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = pixelReader.getColor(x, y);
                double grayscaleValue = color.getRed() * 0.299 + color.getGreen() * 0.587 + color.getBlue() * 0.114;
                matrix[y][x] = 1.0 - grayscaleValue;
            }
        }



        //matrix[y][x] = (pixelReader.getColor(x, y).equals(Color.BLACK)) ? 1 : 0;
        System.out.println("test start");

        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j) {
                System.out.print(BigDecimal.valueOf(matrix[i][j]).setScale(1, RoundingMode.HALF_DOWN) + " ");
            }
            System.out.println();
        }
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        graphicsContext = canvas.getGraphicsContext2D();
        graphicsContext.setStroke(Color.BLACK);
        graphicsContext.setLineWidth(3);
        // Here I can choose the width of the line

        canvas.setOnMousePressed(event -> {
            if(drawingEnabled){
                graphicsContext.beginPath();
                graphicsContext.moveTo(event.getX(), event.getY());
                graphicsContext.setStroke(Color.BLACK);
            }
        });

        canvas.setOnMouseDragged(event -> {
            if(drawingEnabled){
                graphicsContext.lineTo(event.getX(), event.getY());
                graphicsContext.stroke();
            }
        });

        startButton.setOnAction(event -> startDrawing());
        finishButton.setOnAction(event -> finishDrawing());
        clearButton.setOnAction(event -> clearCanvas());
    }
}

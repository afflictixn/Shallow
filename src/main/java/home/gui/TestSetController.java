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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import shallow.Model;
import shallow.ModelInfo;
import shallow.layers.configs.Config;


import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URL;
import java.util.*;

public class TestSetController implements Initializable{

    public final static  int width = 100;
    public final static int height = 100;
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

    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    }



    public void nextButton(){
        // TODO
    }

    public void predictButton(){
        // TODO
    }

    public void evaluateButton(){
        // TODO
    }

    public void refreshLabels(){
        // TODO the same as in EvaluateController
    }

    @FXML
    private AnchorPane innerAnchorPane;

    public void showPicture(double[][] matrix){ // this function shows the picture on the matrix
        GraphicsContext gc = canvas.getGraphicsContext2D();

        for(int i = 0; i<height; i++){
            for(int j = 0; j<width; j++){
                double value = matrix[i][j];
                int grayValue = (int) (value * 255);

                Color color = Color.rgb(grayValue, grayValue, grayValue);
                gc.setFill(color);
                gc.fillRect(i, j, 1, 1);
            }
        }
    }


    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        //innerAnchorPane.setVisible(false); TODO should be uncommented
    }
}

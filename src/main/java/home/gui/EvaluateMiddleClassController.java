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

import java.io.IOException;

public class EvaluateMiddleClassController {
    @FXML
    private Button Return;

    @FXML
    private Button canvas;

    @FXML
    private Button testSet;


    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    }

    public void canvasFunction() throws IOException {
        MainController.getInstance().setBorderPane("Canvas.fxml");
    }

    public void testSetFunction() throws IOException {
        MainController.getInstance().setBorderPane("TestSet.fxml");
    }



}

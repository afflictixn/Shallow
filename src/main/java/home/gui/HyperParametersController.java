package home.gui;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;

import java.io.IOException;

public class HyperParametersController {

    @FXML
    private Button apply;

    @FXML
    private TextField batchSize;

    @FXML
    private TextField epochs;

    @FXML
    private TextField learningRate;

    @FXML
    private Button Return;

    public void Return() throws IOException {
        MainController.getInstance().reset();
    };

}

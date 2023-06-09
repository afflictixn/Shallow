package home.gui;

import javafx.fxml.FXML;
import javafx.scene.control.Button;

import java.io.IOException;

public class LayerController {

    @FXML
    private Button activation;

    @FXML
    private Button convolution;

    @FXML
    private Button dense;

    @FXML
    private Button flatten;

    @FXML
    private Button reshape;

    @FXML
    private Button Return;

    @FXML
    private Button upsampling;

    public void Return() throws IOException {
        MainController.getInstance().reset();
    };

}

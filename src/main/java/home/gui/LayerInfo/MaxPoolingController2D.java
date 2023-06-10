package home.gui.LayerInfo;

import home.gui.MainController;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;

import java.io.IOException;

public class MaxPoolingController2D {
    @FXML
    private Button Return;

    @FXML
    private Button apply;
    @FXML
    private TextField stridesHeight;

    @FXML
    private TextField stridesWidth;

    @FXML
    private TextField kernelHeight;

    @FXML
    private TextField kernelWidth;

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("Layer.fxml");
        // TODO fix this function
    }

    public void ApplyFunction(){
        // TODO bla bla bla
    }

}

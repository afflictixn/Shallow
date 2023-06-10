package home.gui.LayerInfo;

import home.gui.MainController;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.TextField;
import org.w3c.dom.Text;
import shallow.Main;
import shallow.layers.configs.PaddingType;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class ConvolutionController2D implements Initializable {

    @FXML
    private Button Return;

    @FXML
    public Button apply;

    @FXML
    public TextField filters;

    @FXML
    public TextField kernelHeight;

    @FXML
    public TextField kernelWidth;

    @FXML
    public TextField stridesHeight;

    @FXML
    public TextField stridesWidth;

    @FXML
    public ChoiceBox<PaddingType> padding;

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("Layer.fxml");
    }

    public void ApplyFunction(){
        // TODO implement this one
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        padding.getItems().clear();
        padding.getItems().add(PaddingType.SAME);
        padding.getItems().add(PaddingType.VALID);
    }
}

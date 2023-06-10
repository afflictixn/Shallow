package home.gui.LayerInfo;

import home.gui.MainController;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.TextField;
import shallow.layers.weight_init.WeightInitEnum;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class LinearController implements Initializable {

    @FXML
    private Button apply;

    @FXML
    private Button Return;

    @FXML
    private ChoiceBox<WeightInitEnum> box;

    @FXML
    private TextField field;


    public void ApplyFunction(){
        // TODO bla bla bla
    }

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("Layer.fxml");
        // TODO fix this function
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        box.getItems().clear();

        box.getItems().add(WeightInitEnum.HeNormal);
        box.getItems().add(WeightInitEnum.Normal);
        box.getItems().add(WeightInitEnum.ZEROS);
        box.getItems().add(WeightInitEnum.ONES);
        box.getItems().add(WeightInitEnum.HeUniform);
        box.getItems().add(WeightInitEnum.XavierNormal);
        box.getItems().add(WeightInitEnum.XavierUniform);
    }
}

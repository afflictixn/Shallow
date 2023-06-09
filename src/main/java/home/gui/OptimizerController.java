package home.gui;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ComboBox;

import java.io.IOException;

public class OptimizerController {

    @FXML
    private Button apply;

    @FXML
    private ChoiceBox<String> loss;

    @FXML
    private ChoiceBox<String> optimizer;

    @FXML
    private Button Return;

    @FXML
    public void initLossFunctions () {
        loss.getItems().removeAll(loss.getItems());
        loss.getItems().add("Zero");
        loss.getItems().add("HeNormal");
        loss.getItems().add("HeUniform");
        loss.getItems().add("XavierNormal");
        loss.getItems().add("XavierUniform");
    }

    @FXML
    public void initOptimizerFunctions () {
        optimizer.getItems().removeAll(loss.getItems());
        optimizer.getItems().add("Zero");
        optimizer.getItems().add("HeNormal");
        optimizer.getItems().add("HeUniform");
        optimizer.getItems().add("XavierNormal");
        optimizer.getItems().add("XavierUniform");
    }


    public void Return() throws IOException {
        MainController.getInstance().reset();
    };

}

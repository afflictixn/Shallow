package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;

import java.io.IOException;
import java.util.Objects;

public class DatasetController {

    @FXML
    private Button apply;

    @FXML
    private ChoiceBox<String> choiceBox;

    public void Return() throws IOException {
        MainController.getInstance().reset();
    };
    public void Apply(){};

    @FXML
    private Button Return;

    @FXML
    public void initializeDatasets () {
        choiceBox.getItems().removeAll(choiceBox.getItems());
        choiceBox.getItems().add("mnist");
        choiceBox.getItems().add("cifar10");
    }

}

package home.gui;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;

import java.io.IOException;
import java.net.URL;
import java.util.Objects;
import java.util.ResourceBundle;

public class DatasetController implements Initializable {

    @FXML
    private Button apply;

    @FXML
    private Button Return;

    @FXML
    private ChoiceBox<DatasetEnum> choiceBox;


    public void ApplyFunction(){
        // TODO implements this function
    };


//    @FXML
//    public void initializeDatasets () {
//        choiceBox.getItems().clear();
//        choiceBox.getItems().add(DatasetEnum.MNIST);
//        choiceBox.getItems().add(DatasetEnum.CIFAR10);
//    }

    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        choiceBox.getItems().clear();
        choiceBox.getItems().add(DatasetEnum.MNIST);
        choiceBox.getItems().add(DatasetEnum.CIFAR10);

        choiceBox.setOnAction(this::handler);
    }


    public void handler(ActionEvent event){
        System.out.println(choiceBox.getValue());
    }
}

package home.gui;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;

import java.io.IOException;
import java.net.URL;
import java.util.Objects;
import java.util.ResourceBundle;

public class DatasetController implements Initializable {

    @FXML
    private Label resultOfOperation;

    private static DatasetEnum currentDataset = DatasetEnum.CIFAR10;
    // TODO сделать так, чтобы по дефолту этот же датасет был установлен в Connector!!!
    // TODO при нажатии на шторку пропадает надпись

    @FXML
    private Button apply;

    @FXML
    private Button Return;

    @FXML
    private ChoiceBox<DatasetEnum> choiceBox;


    public void ApplyFunction(){
        DatasetEnum temp = choiceBox.getValue();
        currentDataset = temp;
        MainController.getConnector().setDatasetEnum(temp);

        resultOfOperation.setText("Data was successfully applied.");
        resultOfOperation.setTextFill(Color.GREEN);
        resultOfOperation.setVisible(true);

//        System.out.println("done");
//
//        if(!MainController.getConnector().datasetEnum.equals(null)){
//            System.out.println(MainController.getConnector().datasetEnum);
//        }
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
        choiceBox.setValue(currentDataset);

        resultOfOperation.setVisible(false);
        resultOfOperation.setDisable(true);

        //choiceBox.setOnAction(this::handler);
    }


//    public void handler(ActionEvent event){
//        System.out.println(choiceBox.getValue());
//    }
}

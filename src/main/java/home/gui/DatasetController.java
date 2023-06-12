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

    private static DatasetEnum currentDataset = DatasetEnum.MNIST;
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

        resultOfOperation.setTextFill(Color.BLACK);
        resultOfOperation.setText("Data was successfully applied.");
        resultOfOperation.setVisible(true);

//        System.out.println("done");
//
//        if(!MainController.getConnector().datasetEnum.equals(null)){
//            System.out.println(MainController.getConnector().datasetEnum);
//        }
    };



    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    }

    @FXML
    private Label trainSetSize;

    @FXML
    private Label testSetSize;

    @FXML
    private Label inputShape;

    @FXML
    private Label labels;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        choiceBox.getItems().clear();
        choiceBox.getItems().add(DatasetEnum.MNIST);
        choiceBox.getItems().add(DatasetEnum.CIFAR10);
        choiceBox.setValue(currentDataset);

        resultOfOperation.setVisible(false);
        setDescription();
        //resultOfOperation.setDisable(true);

        choiceBox.setOnAction(this::handler);
    }

    public void setDescription(){
        if(choiceBox.getValue().equals(DatasetEnum.MNIST)){
            trainSetSize.setText(MNIST[0]);
            testSetSize.setText(MNIST[1]);
            inputShape.setText(MNIST[2]);
            labels.setText(MNIST[3]);
        }
        else{
            trainSetSize.setText(CIFAR10[0]);
            testSetSize.setText(CIFAR10[1]);
            inputShape.setText(CIFAR10[2]);
            labels.setText(CIFAR10[3]);
        }
    }


    String[] MNIST = {"60000", "10000", "784", "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"};
    String[] CIFAR10 = {"50000", "10000", "32x32x3", "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"};

    private void handler(ActionEvent actionEvent) {
        setDescription();
        resultOfOperation.setVisible(false);
    }


//    public void handler(ActionEvent event){
//        System.out.println(choiceBox.getValue());
//    }
}

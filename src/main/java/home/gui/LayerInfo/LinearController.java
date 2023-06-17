package home.gui.LayerInfo;

import home.gui.MainController;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyEvent;
import javafx.scene.paint.Color;
import shallow.layers.configs.Config;
import shallow.layers.weight_init.WeightInitEnum;
import javafx.event.ActionEvent;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class LinearController implements Initializable {

    @FXML
    private Label resultOfOperation;

    @FXML
    private Button apply;

    @FXML
    private Button Return;

    @FXML
    private ChoiceBox<WeightInitEnum> box1;

    @FXML
    private ChoiceBox<WeightInitEnum> box2;

    @FXML
    private TextField field;


    public void ApplyFunction(){
        int temp = 0;
        String s = field.getText();
        if(s.isEmpty()){
            ++temp;
        }
        if(box1.getValue() == null || box2.getValue() == null){
            ++temp;
        }

        int a = 0;

        try{
            a = Integer.parseInt(s);
            if(a <= 0){
                ++temp;
            }
        }
        catch(Exception e){
            ++temp;
        }

        if(temp == 0){
            MainController.getConnector().processLinear(a, box1.getValue(), box2.getValue());

            List<Config> configs = MainController.getConnector().configs;
            MainController.getInstance().addLayer(configs.get(configs.size() - 1).toString(), configs.get(configs.size() - 1).getDescription());

            resultOfOperation.setText("Data was successfully applied.");
            resultOfOperation.setStyle("-fx-background-color: green");
            resultOfOperation.setVisible(true);
        }
        else{
            resultOfOperation.setText("Entered data is inappropriate.");
            resultOfOperation.setStyle("-fx-background-color: red");
            resultOfOperation.setVisible(true);
        }
    }

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("Layer.fxml");
    }

    public void hideTheLabel(ActionEvent event){
        resultOfOperation.setVisible(false);
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        box1.getItems().clear();
        box2.getItems().clear();

        box1.getItems().add(WeightInitEnum.HeNormal);
        box1.getItems().add(WeightInitEnum.Normal);
        box1.getItems().add(WeightInitEnum.ZEROS);
        box1.getItems().add(WeightInitEnum.ONES);
        box1.getItems().add(WeightInitEnum.HeUniform);
        box1.getItems().add(WeightInitEnum.XavierNormal);
        box1.getItems().add(WeightInitEnum.XavierUniform);

        box2.getItems().add(WeightInitEnum.HeNormal);
        box2.getItems().add(WeightInitEnum.Normal);
        box2.getItems().add(WeightInitEnum.ZEROS);
        box2.getItems().add(WeightInitEnum.ONES);
        box2.getItems().add(WeightInitEnum.HeUniform);
        box2.getItems().add(WeightInitEnum.XavierNormal);
        box2.getItems().add(WeightInitEnum.XavierUniform);

        field.setText("10");
        box1.setValue(WeightInitEnum.HeNormal);
        box2.setValue(WeightInitEnum.ZEROS);
        resultOfOperation.setVisible(false);
        box1.setOnAction(this::hideTheLabel);
        box2.setOnAction(this::hideTheLabel);
    }

    public void hidingTheLabel() {
        resultOfOperation.setVisible(false);
    }
}

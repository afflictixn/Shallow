package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.paint.Color;
import shallow.layers.configs.Config;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class LayerController implements Initializable {

    @FXML
    private Label resultOfOperation;
    private String lastPressedButton = "";
    // last pressed button to use the 'apply' method from this controller

    public void setLastPressedButton(String s){
        this.lastPressedButton = s;
    }

    public String getLastPressedButton(){
        return this.lastPressedButton;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public void showApply(){
        apply.setDisable(false);
    }

    public void hideApply(){
        apply.setDisable(true);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @FXML
    private Button linear;

    @FXML
    private Button flatten;

    @FXML
    private Button convolution2D;
    @FXML
    private Button maxPooling2D;

    @FXML
    private Button reLU;
    @FXML
    private Button sigmoid;

    @FXML
    private Button apply;

    @FXML
    private Button Return;

    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    };

    public void ApplyFunction(){
        if(lastPressedButton.equals("Flatten")){
            MainController.getConnector().processFlatten();

            List<Config> configs = MainController.getConnector().configs;
            MainController.getInstance().addLayer(configs.get(configs.size() - 1).toString(), configs.get(configs.size() - 1).toString() + " Layer");
        }
        else if(lastPressedButton.equals("ReLU")){
            MainController.getConnector().processReLU();

            List<Config> configs = MainController.getConnector().configs;
            MainController.getInstance().addLayer(configs.get(configs.size() - 1).toString(), configs.get(configs.size() - 1).toString() + " Layer");
        }
        else if(lastPressedButton.equals("Sigmoid")){
            MainController.getConnector().processSigmoid();

            List<Config> configs = MainController.getConnector().configs;
            MainController.getInstance().addLayer(configs.get(configs.size() - 1).toString(), configs.get(configs.size() - 1).toString() + " Layer");
        }
        resultOfOperation.setText("Data was successfully applied.");

        resultOfOperation.setVisible(true);
        hideApply();
    }


    public void linearFunction() throws IOException {
        hideApply();
        MainController.getInstance().setBorderPane("LayerInfo/LayerLinear.fxml");
    }

    public void flattenFunction(){
        resultOfOperation.setVisible(false);
        showApply();
        lastPressedButton = "Flatten";
    }

    public void convolution2DFunction() throws IOException {
        hideApply();
        MainController.getInstance().setBorderPane("LayerInfo/LayerConvolution2D.fxml");
    }

    public void maxPooling2DFunction() throws IOException {
        hideApply();
        MainController.getInstance().setBorderPane("LayerInfo/LayerMaxPooling2D.fxml");
    }

    public void reLUFunction(){
        resultOfOperation.setVisible(false);
        showApply();
        lastPressedButton = "ReLU";
    }

    public void sigmoidFunction(){
        resultOfOperation.setVisible(false);
        showApply();
        lastPressedButton = "Sigmoid";
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        hideApply();
        resultOfOperation.setVisible(false);
    }
}

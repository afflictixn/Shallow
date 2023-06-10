package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class LayerController implements Initializable {


    private String lastPressedButton;
    // last pressed button to use the 'apply' method from this controller

    public void setLastPressedButton(String s){
        this.lastPressedButton = s;
    }

    public String getLastPressedButton(){
        return this.lastPressedButton;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public void showApply(){
        apply.setVisible(true);
        apply.setDisable(false);
    }

    public void hideApply(){
        apply.setDisable(true);
        apply.setVisible(false);
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
        //MainController.getInstance().setBorderPane("Basic.fxml");
        // TODO  fix this function
    };

    public void ApplyFunction(){
        hideApply(); //? может быть лучше не убирать
        // TODO bla bla bla
    }


    public void linearFunction() throws IOException {
        hideApply(); // ?
        MainController.getInstance().setBorderPane("LayerInfo/LayerLinear.fxml");
        // TODO fix this function, it should open another scene
    }

    public void flattenFunction(){
        showApply();
        lastPressedButton = "Flatten";
    }

    public void convolution2DFunction() throws IOException {
        hideApply();
        MainController.getInstance().setBorderPane("LayerInfo/LayerConvolution2D.fxml");
        // TODO fix this button, it should open another scene
    }

    public void maxPooling2DFunction() throws IOException {
        hideApply();
        MainController.getInstance().setBorderPane("LayerInfo/LayerMaxPooling2D.fxml");
        // TODO fix this button, it should open another scene
    }

    public void reLUFunction(){
        showApply();
        lastPressedButton = "ReLU";
    }

    public void sigmoidFunction(){
        showApply();
        lastPressedButton = "Sigmoid";
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        hideApply();
    }
}

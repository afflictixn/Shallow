package home.gui.LayerInfo;

import home.gui.MainController;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.paint.Color;
import shallow.layers.configs.Config;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class MaxPoolingController2D implements Initializable {

    @FXML
    private Label resultOfOperation;
    @FXML
    private Button Return;

    @FXML
    private Button apply;
    @FXML
    private TextField stridesHeight;

    @FXML
    private TextField stridesWidth;

    @FXML
    private TextField kernelHeight;

    @FXML
    private TextField kernelWidth;

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("Layer.fxml");
    }

    public void ApplyFunction(){
        int temp = 0;
        String s1 = kernelHeight.getText();
        String s2 = kernelWidth.getText();
        String s3 = stridesHeight.getText();
        String s4 = stridesWidth.getText();
        if(s1.isEmpty() || s2.isEmpty() || s3.isEmpty() || s4.isEmpty()){
            ++temp;
        }
        int i1 = 0; int i2 = 0; int i3 = 0; int i4 = 0;
        try{
            i1 = Integer.parseInt(s1);
            i2 = Integer.parseInt(s2);
            i3 = Integer.parseInt(s3);
            i4 = Integer.parseInt(s4);
            if(i1 <= 0 || i2 <= 0 || i3 <= 0 || i4 <= 0){
                ++temp;
            }
        }
        catch(Exception e){
            ++temp;
        }

        if(temp==0){
            MainController.getConnector().processMaxPool2d(i1, i2, i3, i4);

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

    public void hideTheLabel(){
        resultOfOperation.setVisible(false);
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        stridesHeight.setText("2");
        stridesWidth.setText("2");
        kernelHeight.setText("2");
        kernelWidth.setText("2");


        resultOfOperation.setVisible(false);
    }
}

package home.gui.LayerInfo;

import home.gui.MainController;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.paint.Color;
import org.w3c.dom.Text;
import shallow.Main;
import shallow.layers.configs.PaddingType;
import shallow.layers.weight_init.WeightInitEnum;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class ConvolutionController2D implements Initializable {
    @FXML
    private Label resultOfOperation;
    @FXML
    private Button Return;

    @FXML
    public Button apply;

    @FXML
    public TextField filters;

    @FXML
    public TextField kernelHeight;

    @FXML
    public TextField kernelWidth;

    @FXML
    public TextField stridesHeight;

    @FXML
    public TextField stridesWidth;

    @FXML
    public ChoiceBox<PaddingType> padding;

    @FXML
    public ChoiceBox<WeightInitEnum> weight;

    @FXML
    public ChoiceBox<WeightInitEnum> bias;

    public void ReturnFunction() throws IOException {
        MainController.getInstance().setBorderPane("Layer.fxml");
    }

    public void ApplyFunction(){
        int temp = 0;
        String s1 = filters.getText();
        String s2 = kernelHeight.getText();
        String s3 = kernelWidth.getText();
        String s4 = stridesHeight.getText();
        String s5 = stridesWidth.getText();
        if(s1.isEmpty() || s2.isEmpty() || s3.isEmpty() || s4.isEmpty() || s5.isEmpty()){
            ++temp;
        }
        if(padding.getValue() == null || weight.getValue() == null || bias.getValue() == null){
            ++temp;
        }
        int i1 =0;
        int i2 = 0;
        int i3 = 0;
        int i4 = 0;
        int i5 = 0;
        try{
            i1 = Integer.parseInt(s1);
            i2 = Integer.parseInt(s2);
            i3 = Integer.parseInt(s3);
            i4 = Integer.parseInt(s4);
            i5 = Integer.parseInt(s5);
            if(i1 <= 0 || i2 <= 0 || i3 <= 0 || i4 <= 0 || i5 <= 0){
                ++temp;
            }
        }
        catch(Exception e){
            ++temp;
        }

        if(temp==0){
            MainController.getConnector().processConv2d(i1, i2, i3, i4, i5, padding.getValue(), weight.getValue(), bias.getValue());
            // TODO fix this problem
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

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        padding.getItems().clear();
        padding.getItems().add(PaddingType.SAME);
        padding.getItems().add(PaddingType.VALID);

        weight.getItems().clear();
        bias.getItems().clear();

        weight.getItems().add(WeightInitEnum.HeNormal);
        weight.getItems().add(WeightInitEnum.Normal);
        weight.getItems().add(WeightInitEnum.ZEROS);
        weight.getItems().add(WeightInitEnum.ONES);
        weight.getItems().add(WeightInitEnum.HeUniform);
        weight.getItems().add(WeightInitEnum.XavierNormal);
        weight.getItems().add(WeightInitEnum.XavierUniform);

        bias.getItems().add(WeightInitEnum.HeNormal);
        bias.getItems().add(WeightInitEnum.Normal);
        bias.getItems().add(WeightInitEnum.ZEROS);
        bias.getItems().add(WeightInitEnum.ONES);
        bias.getItems().add(WeightInitEnum.HeUniform);
        bias.getItems().add(WeightInitEnum.XavierNormal);
        bias.getItems().add(WeightInitEnum.XavierUniform);

        filters.setText("5");
        kernelHeight.setText("3");
        kernelWidth.setText("3");
        stridesWidth.setText("1");
        stridesHeight.setText("1");
        padding.setValue(PaddingType.SAME);
        weight.setValue(WeightInitEnum.HeNormal);
        bias.setValue(WeightInitEnum.ZEROS);

        resultOfOperation.setVisible(false);
    }
}

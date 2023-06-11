package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.paint.Color;
import shallow.Main;
import shallow.layers.weight_init.WeightInitEnum;
import shallow.losses.LossEnum;
import shallow.optimizers.OptimizerEnum;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class OptimizerController implements Initializable {
    // TODO сделать так чтобы мои начальные значени совпадали со значениями в коннекторе
    // TODO сделать так, чтобы Label пропадал, когда он уже не нужен
    // TODO проверить предыдущий пункт для остальных контроллеров

    @FXML
    private Label resultOfOperation;

    private static double lastBeta1 = 1.0;
    private static double lastBeta2 = 1.0;
    private static double lastMomentum = 1.0;

    private static LossEnum lastLoss = LossEnum.BinaryCrossEntropyLoss;
    private static OptimizerEnum lastOptimizer = OptimizerEnum.Adam;

    @FXML
    private Button apply;

    @FXML
    private Button Return;

    @FXML
    private ChoiceBox<LossEnum> lossBox;

    @FXML
    private ChoiceBox<OptimizerEnum> optimizerBox;

    @FXML
    private Label beta1;

    @FXML
    private Label beta2;

    @FXML
    private Label momentum;

    @FXML
    private TextField beta1Field;

    @FXML
    private TextField beta2Field;

    @FXML
    private TextField momentumField;

    public void showAdam(){
        beta1.setDisable(false);
        beta2.setDisable(false);
        beta1Field.setDisable(false);
        beta2Field.setDisable(false);
//        momentum.setDisable(true);
//        momentumField.setDisable(true);

        beta1.setVisible(true);
        beta2.setVisible(true);
        beta1Field.setVisible(true);
        beta2Field.setVisible(true);
        momentum.setVisible(false);
        momentumField.setVisible(false);
    }

    public void showSGD(){
//        beta1.setDisable(true);
//        beta2.setDisable(true);
//        beta1Field.setDisable(true);
//        beta2Field.setDisable(true);
        momentum.setDisable(false);
        momentumField.setDisable(false);

        beta1.setVisible(false);
        beta2.setVisible(false);
        beta1Field.setVisible(false);
        beta2Field.setVisible(false);
        momentum.setVisible(true);
        momentumField.setVisible(true);
    }


    public void handleTheChoice(){
        OptimizerEnum o = optimizerBox.getValue();
        if(o.equals(OptimizerEnum.Adam)){
            showAdam();
        }
        else{
            showSGD();
        }
    }


    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    };

    public void ApplyFunction(){
        if(optimizerBox.getValue().equals(OptimizerEnum.Adam)){
            ApplyAdam();
        }
        else{
            ApplySGD();
        }
    }

    public void ApplyAdam(){
        int temp = 0;
        String s1 = beta1Field.getText();
        String s2 = beta2Field.getText();
        if(s1.isEmpty() || s2.isEmpty()){
            ++temp;
        }
        double i1 = 0;
        double i2 = 0;
        try{
            i1 = Double.parseDouble(s1);
            i2 = Double.parseDouble(s2);
            if(i1 <= 0 || i2 <= 0){
                ++temp;
            }
        }
        catch (Exception e){
            ++temp;
        }

        if(temp == 0){
            MainController.getConnector().setOptimizerAdam(i1, i2);
            MainController.getConnector().setLossEnum(lossBox.getValue());
            lastBeta1 = i1;
            lastBeta2 = i2;

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

    public void ApplySGD(){
        System.out.println("applying");
        int temp = 0;
        String s = momentumField.getText();
        if(s.isEmpty()){
            System.out.println("empty");
            ++temp;
        }
        double i = 0;
        try{
            i = Double.parseDouble(s);
            if(i <= 0){
                System.out.println("less than zero");
                ++temp;
            }
        }
        catch (Exception e){
            System.out.println("mistake");
            ++temp;
        }

        if(temp == 0){
            MainController.getConnector().setOptimizerSGD(i);
            MainController.getConnector().setLossEnum(lossBox.getValue());
            lastMomentum = i;

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
        lossBox.getItems().clear();
        lossBox.getItems().add(LossEnum.BinaryCrossEntropyLoss);
        lossBox.getItems().add(LossEnum.CategoricalCrossEntropyLoss);

        optimizerBox.getItems().clear();
        optimizerBox.getItems().add(OptimizerEnum.Adam);
        optimizerBox.getItems().add(OptimizerEnum.StochasticGradientDescent);

        beta1Field.setText(Double.toString(lastBeta1));
        beta2Field.setText(Double.toString(lastBeta2));
        momentumField.setText(Double.toString(lastMomentum));

        lossBox.setValue(lastLoss);
        optimizerBox.setValue(lastOptimizer);

        if(optimizerBox.getValue().equals(OptimizerEnum.Adam)){
            showAdam();
        }
        else{
            showSGD();
        }

        resultOfOperation.setVisible(false);

        optimizerBox.setOnAction(event -> handleTheChoice());
        // TODO сделать чтобы снчала labels // TextFields были невидимыми
    }
}

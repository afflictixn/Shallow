package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import shallow.layers.weight_init.WeightInitEnum;
import shallow.losses.LossEnum;
import shallow.optimizers.OptimizerEnum;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class OptimizerController implements Initializable {

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



    public void handleTheChoice(){
        OptimizerEnum o = optimizerBox.getValue();
        if(o.equals(OptimizerEnum.Adam)){
            beta1.setDisable(false);
            beta2.setDisable(false);
            beta1Field.setDisable(false);
            beta2Field.setDisable(false);
            momentum.setDisable(true);
            momentumField.setDisable(true);

            beta1.setVisible(true);
            beta2.setVisible(true);
            beta1Field.setVisible(true);
            beta2Field.setVisible(true);
            momentum.setVisible(false);
            momentumField.setVisible(false);
        }
        else{
            beta1.setDisable(true);
            beta2.setDisable(true);
            beta1Field.setDisable(true);
            beta2Field.setDisable(true);
            momentum.setDisable(false);
            momentumField.setDisable(false);

            beta1.setVisible(false);
            beta2.setVisible(false);
            beta1Field.setVisible(false);
            beta2Field.setVisible(false);
            momentum.setVisible(true);
            momentumField.setVisible(true);
        }
    }


    public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
    };

    public void ApplyFunction(){
        // TODO implement this function
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        lossBox.getItems().clear();
        lossBox.getItems().add(LossEnum.BinaryCrossEntropyLoss);
        lossBox.getItems().add(LossEnum.CategoricalCrossEntropyLoss);

        optimizerBox.getItems().clear();
        optimizerBox.getItems().add(OptimizerEnum.Adam);
        optimizerBox.getItems().add(OptimizerEnum.StochasticGradientDescent);

        optimizerBox.setOnAction(event -> handleTheChoice());
        // TODO сделать чтобы снчала labels // TextFields были невидимыми
    }
}

package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.paint.Color;

import java.io.IOException;
import java.math.BigDecimal;
import java.net.URL;
import java.util.ResourceBundle;

public class HyperParametersController implements Initializable {

    private static int lastBatchSize = 64;
    private static int lastEpochs = 20;
    private static double lastLearningRate = 0.05;

    private static double lastL2Regularization = 0.0005;

    @FXML
    private Label resultOfOperation;

    @FXML
    private Button apply;

    @FXML
    private TextField batchSize;

    @FXML
    private TextField epochs;

    @FXML
    private TextField learningRate;

    @FXML
    private TextField l2Regularization;

    @FXML
    private Button Return;

   public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
   }

   public void hideTheLabel(){
       resultOfOperation.setVisible(false);
   }

   public void ApplyFunction(){
       int temp = 0;
       String s1 = batchSize.getText();
       String s2 = epochs.getText();
       String s3 = learningRate.getText();
       String s4 = l2Regularization.getText();

       if(s1.isEmpty() || s2.isEmpty() || s3.isEmpty() || s4.isEmpty()){
           ++temp;
       }
       int i1 = 0;
       int i2 = 0;
       double i3 = 0;
       double i4 = 0;
       try{
           i1 = Integer.parseInt(s1);
           i2 = Integer.parseInt(s2);
           i3 = Double.parseDouble(s3);
           i4 = Double.parseDouble(s4);

           if(i1 <= 0 || i2 <= 0 || i3 <= 0 || i4<0){
               ++temp;
           }
       }
       catch(Exception e){
           ++temp;
       }

       if(temp == 0){
           lastBatchSize = i1;
           lastEpochs = i2;
           lastLearningRate = i3;
           lastL2Regularization = i4;
           MainController.getConnector().setHyperParametersInfo(i1, i2, i3, i4);
           MainController.currentLearningRate.set(i3);

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
       resultOfOperation.setVisible(false);

       batchSize.setText(Integer.toString(lastBatchSize));
       epochs.setText(Integer.toString(lastEpochs));
       learningRate.setText(Double.toString(lastLearningRate));
       l2Regularization.setText(BigDecimal.valueOf(lastL2Regularization).toString());
    }
}

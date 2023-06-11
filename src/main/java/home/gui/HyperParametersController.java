package home.gui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.paint.Color;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class HyperParametersController implements Initializable {

    private static int lastBatchSize = 1;
    private static int lastEpochs = 1;
    private static double lastLearningRate = 1.0;

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
    private Button Return;

   public void ReturnFunction() throws IOException {
        MainController.getInstance().reset();
   }

   public void ApplyFunction(){
       int temp = 0;
       String s1 = batchSize.getText();
       String s2 = epochs.getText();
       String s3 = learningRate.getText();

       if(s1.isEmpty() || s2.isEmpty() || s3.isEmpty()){
           ++temp;
       }
       int i1 = 0;
       int i2 = 0;
       double i3 = 0;
       try{
           i1 = Integer.parseInt(s1);
           i2 = Integer.parseInt(s2);
           i3 = Double.parseDouble(s3);

           if(i1 <= 0 || i2 <= 0 || i3 <= 0){
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
           MainController.getConnector().setHyperParametersInfo(i1, i2, i3);

           resultOfOperation.setText("Data was successfully applied.");
           resultOfOperation.setTextFill(Color.GREEN);
           resultOfOperation.setVisible(true);
       }
       else{
           resultOfOperation.setText("Entered data is inappropriate.");
           resultOfOperation.setTextFill(Color.RED);
           resultOfOperation.setVisible(true);
       }
   }


    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
       resultOfOperation.setVisible(false);

       batchSize.setText(Integer.toString(lastBatchSize));
       epochs.setText(Integer.toString(lastEpochs));
       learningRate.setText(Double.toString(lastLearningRate));
    }
}

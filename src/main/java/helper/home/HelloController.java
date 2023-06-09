package helper.home;

import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

public class HelloController {
    // --- Main buttons, all of them are located on the left side of the screen

    @FXML
    private Label checker;
    int ch = 0;
    public void checker(){
        ++ch;
        if(ch%2==1){
            checker.setText("odd");
        }
        else{
            checker.setText("even");
        }
    }

    public static class Information{
        public String NameOfDataset;
        public int NumberOfLayers;
    }
    Information information = new Information();
    @FXML
    private VBox VBoxOptions;
    public void OptionsClear(){
        for(Node node : Options.getChildren()){
            node.setVisible(false);
            node.setDisable(true);
        }
        Options.setVisible(false);
        Options.setDisable(true);
    }
    @FXML
    private Button Dataset;
    @FXML
    private Button NewLayer;
    @FXML
    private Button Parameters;
    @FXML
    private Button Optimizer;
    @FXML
    private Button Evaluate;
    @FXML
    private Button Graph;
    //

    // ---- Anchor pane where all the options will be held
    @FXML
    private AnchorPane Options;
    //

    // -----
    // this buttons I will be always using in our program, so i will leave them here

    @FXML
    private ImageView Ok;


    ///////////////////////////////////////////////////////////////////////////////////////////

    public void Dataset(){


    }
    /////////////

    @FXML
    private Button DatasetBack;
    @FXML
    private Button DatasetApply;

    public void DatasetBack(){
        DatasetOpen = false;

        for(Node node : Options.getChildren()){

            node.setVisible(false);
            node.setDisable(true);

        }

        /// TODO send the signal to all of the left buttons, that there children to the right are closed
//        Ok.setVisible(false);
        Options.setVisible(false);
        Options.setDisable(true);
        Dataset.setStyle("-fx-border-color: #7156DD;");
    }

    public void DatasetApply(){
        Ok.setVisible(true);
        /// TODO send some message to the class

        // Layer.setValue(text.getText() --> );
        //Main.begin();
    }
    boolean DatasetOpen = false;
    @FXML
    private VBox DatasetBox;
    @FXML
    private Text selectDataset;
    @FXML
    private Button first;
    @FXML
    private Button second;
    @FXML
    private Button third;
    @FXML
    private Button fourth;
    public void First(){
        Ok.setVisible(false);
        ButtonsDefaultColor();
        first.setStyle("-fx-border-color: #3F2B63;");
        information.NameOfDataset = "first";
        //first.setStyle("-fx-background-color: red;");

    }
    public void Second(){
        Ok.setVisible(false);
        ButtonsDefaultColor();
        second.setStyle("-fx-border-color: #3F2B63;");
        information.NameOfDataset = "second";
        //second.setStyle("-fx-background-color: red;");
    }
    public void Third(){
        Ok.setVisible(false);
        ButtonsDefaultColor();
        third.setStyle("-fx-border-color: #3F2B63;");
        information.NameOfDataset = "third";
    }
    public void Fourth(){
        Ok.setVisible(false);
        ButtonsDefaultColor();
        fourth.setStyle("-fx-border-color: #3F2B63;");
        information.NameOfDataset = "fourth";
    }

    public void ButtonsDefaultColor(){
        first.setStyle("-fx-border-color: #7156DD;");
        second.setStyle("-fx-border-color: #7156DD;");
        third.setStyle("-fx-border-color: #7156DD;");
        fourth.setStyle("-fx-border-color: #7156DD;");
    }


    @FXML
    private ScrollPane choose_layer;
    @FXML
    private ScrollPane Linear_layer_params;
    @FXML
    private Button back_to_layer_menu;
    @FXML
    private TextField linear_units;
    @FXML
    public ChoiceBox<String> linear_weight_choicebox;
    @FXML
    public ChoiceBox<String> linear_bias_choicebox;
    @FXML
    private Button add_linear_button;
    @FXML
    private ScrollPane choose_dataset;
    @FXML
    private ChoiceBox<String> dataset_choicebox;
    @FXML
    private Button load_dataset_button;

    public void initialize_layers() {
        linear_weight_choicebox.setItems(FXCollections.observableArrayList("Zero", "HeNormal", "HeUniform",
                "XavierNormal", "XavierNormal"));
        linear_bias_choicebox.setItems(FXCollections.observableArrayList("Zero", "HeNormal", "HeUniform",
                "XavierNormal", "XavierNormal"));
        linear_units.setText("32");
    }

    private void clear_layer_menu(){
        choose_layer.setVisible(false);
        Linear_layer_params.setVisible(false);
        back_to_layer_menu.setVisible(false);
        add_linear_button.setVisible(false);
    }
    private void clear_dataset_menu(){
        choose_dataset.setVisible(false);
    }
    private void clear_menu(){
        clear_layer_menu();
        clear_dataset_menu();
    }

    public void NewLayer() {
        if (!choose_layer.isVisible()) {
            clear_menu();
            choose_layer.setVisible(true);
        } else {
            clear_layer_menu();
        }
    }

    public void Linear_layer() {
        initialize_layers();
        add_linear_button.setVisible(true);
        Linear_layer_params.setVisible(true);
        back_to_layer_menu.setVisible(true);
    }

    public void Back_to_layer_menu() {
        back_to_layer_menu.setVisible(false);
        clear_layer_menu();
        choose_layer.setVisible(true);
    }


    public void show_dataset_menu() {
        if (!choose_dataset.isVisible()) {
            clear_menu();
            choose_dataset.setVisible(true);
            dataset_choicebox.setItems(FXCollections.observableArrayList("MNIST", "CIFAR10", "CIFAR100"));
        } else {
            clear_dataset_menu();
        }
    }

    @FXML
    private ImageView LayerOK1;
    @FXML
    private Text text1;
    @FXML
    private Text text2;
    @FXML
    private Text text3;
    @FXML
    private Button LayerApply1;
    public void LayerApply1(){
        String s = LayerEnter.getText();
        try{
            int a = Integer.parseInt(s);
            /// TODO send this information to some other place
            text3.setVisible(false);
            LayerOK1.setVisible(true);
        }
        catch(NumberFormatException e){
//            LayerEnter.setText("");
            text3.setVisible(true);
            return;
        }
    }
    private String current_layer = "Layer1";

    @FXML
    private TextField LayerEnter;
    public void LayerEnter(){
        LayerOK1.setVisible(false);
    }
    @FXML
    private VBox LayerVBox;
    @FXML
    private Button Layer1;
    public void Layer1(){
        current_layer = "Layer1";
        LayerOK1.setVisible(false);
    }
    @FXML
    private Button Layer2;
    public void Layer2(){
        current_layer = "Layer2";
        LayerOK1.setVisible(false);
    }
    @FXML
    private Button Layer3;
    public void Layer3(){
        current_layer = "Layer3";
        LayerOK1.setVisible(false);
    }



    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    public void Parameters(){

    }

    private boolean ParametersOpen = false;
    @FXML
    private AnchorPane ParametersPane;
    @FXML
    private Text Batch;
    @FXML
    private Text Epochs;
    @FXML
    private Text LearningRate;
    @FXML
    private Button ParametersBack;
    @FXML
    private Button ParametersApply;
    @FXML
    private Text ParametersError;
    @FXML
    private TextField field1;
    @FXML
    private TextField field2;
    @FXML
    private TextField field3;

    public void ParametersApply(){
        int a = -1;
        int b = -1;
        double c = -1;

    }







}
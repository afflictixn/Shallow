package home.gui;

import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import shallow.Model;
import shallow.ModelInfo;
import shallow.layers.configs.Config;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URL;
import java.util.*;

public class MainController implements Initializable {
    public static AtomicDouble currentLearningRate = new AtomicDouble(0);
    private static Connector connector;
    public static ModelInfo modelInfo;
    public static Model neuralNetworkModel;

    ArrayList<Button> ListOfButtons = new ArrayList<>();

    @FXML
    private VBox architecture;

    @FXML
    private Button temp;

    HashMap<Button, String> information = new HashMap<>();

    @FXML
    private Label informationLabel;

    @FXML
    private Button removeLayer;

    public void addLayer(String name, String info){
        Button button = createButton(name);
        button.setOnAction(this::showInformation);
        information.put(button, info);
        ListOfButtons.add(button);
        architecture.getChildren().add(button);
        if(ListOfButtons.size()==1){
            removeLayer.setDisable(false);
        }
    }
    private Button createButton(String name){
        Button button = new Button(name);
        button.setPrefHeight(40);
        button.setPrefWidth(300);
        button.setStyle("-fx-background-color: #52548e; -fx-text-fill: white;");
        button.setOnMouseEntered(event -> {
            button.setStyle("-fx-background-color: #8a8cee; -fx-text-fill: white;");
        });
        button.setOnMouseExited(event -> {
            button.setStyle("-fx-background-color: #52548e; -fx-text-fill: white;");
        });
        return button;
    }

    public void showInformation(ActionEvent event){
        Button clickedButton = (Button) event.getSource();
        informationLabel.setText(information.get(clickedButton));
    }

    public void removeLayerFunction(){
        informationLabel.setText(""); //
        Button button = ListOfButtons.get(ListOfButtons.size() - 1);
        information.remove(button);
        List<Config> configs = MainController.getConnector().configs;
        configs.remove(configs.get(configs.size() - 1));
        ListOfButtons.remove(ListOfButtons.size() - 1);
        architecture.getChildren().remove(architecture.getChildren().size()-1);
        if(ListOfButtons.isEmpty()){
            removeLayer.setDisable(true);
        }
    }




    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    public MainController() {
        instance = this;
        connector = new Connector();
    }

    public static MainController getInstance() {
        return instance;
    }

    public static Connector getConnector() {
        return connector;
    }

    public void reset() throws IOException {
        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("Basic.fxml")));
        center.setCenter(p);
    }


    public void setBorderPane(String s) throws IOException {
        if (center == null) {
            System.out.println("null");
        }
        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource(s)));
        center.setCenter(p);
    }

    private static MainController instance;

    @FXML
    private VBox box;

    @FXML
    public BorderPane center;

    @FXML
    private Button dataset;

    @FXML
    private Button evaluate;

    @FXML
    private Button hyperparameters;

    @FXML
    private Button layer;

    @FXML
    private Button optimizer;

    public void datasetFunction() throws IOException {
        getInstance().setBorderPane("Dataset.fxml");
    }

    public void layerFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("Layer.fxml");
    }

    public void optimizerFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("Optimizer.fxml");
    }

    public void hyperparametersFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("HyperParameters.fxml");
    }

    public void evaluateFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("EvaluateMiddleClass.fxml");
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Model buildModel(ModelInfo modelInfo) {
        Model model = new Model(modelInfo);
        if (connector.datasetEnum.equals(DatasetEnum.CIFAR10)) {
            model.isNCHWOrder = true;
        }
        for (Config config : connector.configs) {
            model.addLayer(config.buildLayer());
        }
        // set learning rate scheduler that will react on user input during training
        model
                .setScheduler((initialRate, epochNum) -> currentLearningRate.get())
                .setOptimizer(connector.optimizer)
                .setLoss(connector.lossEnum.getLoss())
                .setL2Regularization(connector.hyperParametersInfo.L2RegularizationLambda);
        return model;
    }

    @FXML
    private Button startButton; // green button to start the training

    @FXML
    private Button stopButton; // red button to stop the training process
    public void showStop(){
        startButton.setVisible(false);
        startButton.setDisable(true);

        stopButton.setDisable(false);
        stopButton.setVisible(true);
    }
    public void showStart(){
        stopButton.setVisible(false);
        stopButton.setDisable(true);

        startButton.setDisable(false);
        startButton.setVisible(true);
    }
    public void startButton() throws IOException { // starts the training of the model
        removeLayer.setDisable(true);
        int batchSize = connector.hyperParametersInfo.batchSize;
        String totalEpochString = String.valueOf(connector.hyperParametersInfo.epochs);

        epochDisplayValue.setText(1 + " of " + totalEpochString);
        int seed = new Random().nextInt();

        // building model for training
        DataSetIterator trainIterator = connector.datasetEnum.getTrainDataSetIterator(batchSize, seed);
        modelInfo = new ModelInfo(connector.hyperParametersInfo.epochs);
        neuralNetworkModel = buildModel(modelInfo);
        Runnable runTrainModel = new Runnable() {
            @Override
            public void run() {
                switch (connector.datasetEnum) {
                    case MNIST -> {
                        neuralNetworkModel.fit(trainIterator,
                                connector.hyperParametersInfo.learningRate,
                                connector.hyperParametersInfo.epochs);
                    }
                    case CIFAR10 -> {
                        DataSet trainData = trainIterator.next();
                        INDArray trainFeatures = trainData.getFeatures().div(255);
                        neuralNetworkModel.fit(trainFeatures, trainData.getLabels(), connector.hyperParametersInfo.learningRate,
                                connector.hyperParametersInfo.batchSize, connector.hyperParametersInfo.epochs);
                    }
                }
                showStart();
                removeLayer.setDisable(false);
            }
        };
        Task<Integer> displayInfoTask = new Task<Integer>() {
            @Override
            protected Integer call() throws Exception {
                while (true) {
                    try {
                        synchronized (modelInfo) {
                            modelInfo.wait();
                        }
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    updateValue(modelInfo.getCurrentEpoch());
                    if (modelInfo.getCurrentEpoch() == modelInfo.getTotalEpoch() || modelInfo.isStopTraining()) {
                        break;
                    }
                }
                return null;
            }
        };
        displayInfoTask.valueProperty().addListener((observableValue, oldValue, newValue) -> {
            lossDisplayValue.setText(String.valueOf(BigDecimal.valueOf(modelInfo.getCurrentLoss())
                    .setScale(2, RoundingMode.HALF_UP)));
            accuracyDisplayValue.setText(BigDecimal.valueOf(modelInfo.accuracy() * 100)
                    .setScale(2, RoundingMode.HALF_UP) + " %");
            epochDisplayValue.setText(modelInfo.getCurrentEpoch() + " of " + totalEpochString);
        });


        Thread trainModel = new Thread(runTrainModel, "trainModel");
        trainModel.setDaemon(true);
        Thread displayInfo = new Thread(displayInfoTask);
        displayInfo.setDaemon(true);
        displayInfo.start();

        long startTime = System.currentTimeMillis();
        trainModel.start();
        // run a thread to display time passed since the beginning of training
        Task<Void> updateTimeTask = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                while (true) {
                    long passedTime = (System.currentTimeMillis() - startTime) / 1000;
                    updateMessage(passedTime + " s");
                    Thread.sleep(1000);
                    if (modelInfo.getCurrentEpoch() == modelInfo.getTotalEpoch() || modelInfo.isStopTraining()) {
                        break;
                    }
                }
                return null;
            }
        };

        showStop();
        Thread displayTime = new Thread(updateTimeTask);
        displayTime.setDaemon(true);
        displayTime.start();
        timeDisplayValue.textProperty().bind(updateTimeTask.messageProperty());
    }

    public void stopButton() {
        modelInfo.setStopTraining(true);

        stopButton.setDisable(true);
        stopButton.setVisible(false);
        startButton.setDisable(false);
        startButton.setVisible(true);
    }

    @FXML
    private Label lossDisplayValue;

    @FXML
    private Label accuracyDisplayValue;

    @FXML
    private Label epochDisplayValue;

    @FXML
    private Label timeDisplayValue;


    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        try {
            getInstance().setBorderPane("Basic.fxml");
        } catch (Exception e) {
            System.out.println("exception");
        }

        stopButton.setVisible(false);
        stopButton.setDisable(true);

        informationLabel.setText("");
        if(ListOfButtons.isEmpty()){
            removeLayer.setDisable(true);
        }
        else{
            removeLayer.setDisable(false);
        }

    }
}

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
    public static Model neuralNetworkModel;

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
    // this is an effective way to switch between scenes
    @FXML
    private VBox box;

    @FXML
    public BorderPane center;

    @FXML
    private Button dataset;

    @FXML
    private Button Evaluate;

    @FXML
    private Button testSet;

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
        getInstance().setBorderPane("Evaluater.fxml");
    }

    public void testSetFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("TestSet.fxml");
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @FXML
    private Button startButton; // green button to start the training

    @FXML
    private Button stopButton; // red button to stop the training process

    public void startButton() throws IOException { // starts the training of the model, actually button is green :)
        int batchSize = connector.hyperParametersInfo.batchSize;
        String totalEpochString = String.valueOf(connector.hyperParametersInfo.epochs);

        epochDisplayValue.setText(1 + " of " + totalEpochString);
        int seed = new Random().nextInt();

        // building model for training
        DataSetIterator trainIterator = connector.datasetEnum.getTrainDataSetIterator(batchSize, seed);
//        DataSetIterator testIterator = connector.datasetEnum.getTestDataSetIterator(batchSize, seed);
        ModelInfo modelInfo = new ModelInfo(connector.hyperParametersInfo.epochs);
        neuralNetworkModel = new Model(modelInfo);
        if(connector.datasetEnum.equals(DatasetEnum.CIFAR10)){
            neuralNetworkModel.isNCHWOrder = true;
        }
        for (Config config : connector.configs) {
            neuralNetworkModel.addLayer(config.buildLayer());
        }
        // set learning rate scheduler that will react on user input during training
        neuralNetworkModel.setScheduler((initialRate, epochNum) -> currentLearningRate.get());
        neuralNetworkModel.setOptimizer(connector.optimizer);
        neuralNetworkModel.setLoss(connector.lossEnum.getLoss());

        Runnable runTrainModel = new Runnable() {
            @Override
            public void run() {
                neuralNetworkModel.fit(trainIterator,
                        connector.hyperParametersInfo.learningRate,
                        connector.hyperParametersInfo.epochs);
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
                    if (modelInfo.getCurrentEpoch() == modelInfo.getTotalEpoch()) {
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
        Task<String> updateTimeTask = new Task<String>() {
            @Override
            protected String call() throws Exception {
                while (true) {
                    long passedTime = (System.currentTimeMillis() - startTime) / 1000;
                    updateMessage(passedTime + " s");
                    Thread.sleep(1000);
                    if (modelInfo.getCurrentEpoch() == modelInfo.getTotalEpoch()) {
                        break;
                    }
                }
                return null;
            }
        };
        startButton.setVisible(false);
        startButton.setDisable(true);

        stopButton.setDisable(false);
        stopButton.setVisible(true);
        Thread displayTime = new Thread(updateTimeTask);
        displayTime.setDaemon(true);
        displayTime.start();
        timeDisplayValue.textProperty().bind(updateTimeTask.messageProperty());

    }

    public void stopButton(){
        // TODO write here what you want to do when the stopButton is pressed
        // TODO also should deactivate the stop button when the process in finished




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

    }
}

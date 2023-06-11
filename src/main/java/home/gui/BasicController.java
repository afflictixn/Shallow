package home.gui;

import com.gluonhq.attach.util.Platform;
import javafx.concurrent.Task;
import javafx.concurrent.Worker;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import shallow.ModelInfo;
import shallow.Model;
import shallow.layers.configs.Config;

import java.io.IOException;
import java.util.Random;

import static javafx.application.Platform.runLater;

public class BasicController {
    private static BasicController instance;
    public static boolean needUpdate = false;
    public BasicController(){
        instance = this;
    }
    public static BasicController getInstance(){
        return instance;
    }
    @FXML
    public Label label;
    int i = 0;

    @FXML
    private Button button;

    public void Func() throws IOException {
        needUpdate = true;
        ++i;
        label.setText("current value : " + i);

        Connector connector = MainController.getConnector();

        int batchSize = connector.hyperParametersInfo.batchSize;
        int seed = new Random().nextInt();
        DataSetIterator trainIterator = connector.datasetEnum.getTrainDataSetIterator(batchSize, seed);
        DataSetIterator testIterator = connector.datasetEnum.getTestDataSetIterator(batchSize, seed);
        ModelInfo modelInfo = new ModelInfo();
        Model model = new Model(modelInfo);
        for(Config config : connector.configs){
            model.addLayer(config.buildLayer());
        }
        model.setOptimizer(connector.optimizer);
        model.setLoss(connector.lossEnum.getLoss());

        Runnable runTrainModel = new Runnable() {
            @Override
            public void run() {
                model.fit(trainIterator,
                        connector.hyperParametersInfo.learningRate,
                        connector.hyperParametersInfo.epochs);
            }
        };
        Task<Void> runDisplayInfo = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                while (true) {
                    try {
                        synchronized (modelInfo) {
                            modelInfo.wait();
                        }
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    updateMessage(modelInfo.toString());
                }
            }
        };
        Thread trainModel = new Thread(runTrainModel, "trainModel");
        trainModel.setDaemon(true);
        label.textProperty().bind(runDisplayInfo.messageProperty());
        Thread displayInfo = new Thread(runDisplayInfo);
        displayInfo.start();
        trainModel.start();

    }


}

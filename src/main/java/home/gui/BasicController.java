package home.gui;

import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import shallow.ModelInfo;
import shallow.Model;
import shallow.layers.configs.Config;

import java.io.IOException;
import java.util.Random;

public class BasicController {
    private static BasicController instance;
    public BasicController(){
        instance = this;
    }
    public static BasicController getInstance(){
        return instance;
    }



}

package home.gui;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Button;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;

import java.io.IOException;
import java.util.Objects;

public class MainController {

    private static Connector connector;


    public MainController(){
        instance = this;
        connector = new Connector();
        //reset();
    }

    public static MainController getInstance(){
        return instance;
    }

    public static Connector getConnector(){
        return connector;
    }

    public void reset() throws IOException {
        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("Basic.fxml")));
        // В целом можно было бы придумать что-то другое, но нам нормально
        center.setCenter(p);
    }


    public void setBorderPane(String s) throws IOException {
        if(center == null){
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
    private Button evaluate;

    @FXML
    private Button hyperparameters;

    @FXML
    private Button layer;

    @FXML
    private Button optimizer;

    public void datasetFunction() throws IOException {
        //DatasetController.what = center;
        getInstance().setBorderPane("Dataset.fxml");
        //AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("Dataset.fxml")));
        //center.setCenter(p);
    }

    public void layerFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("Layer.fxml");
//        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("Layer.fxml")));
//        center.setCenter(p);
    }

    public void optimizerFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("Optimizer.fxml");
//        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("Optimizer.fxml")));
//        center.setCenter(p);
    }

    public void hyperparametersFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("HyperParameters.fxml");
//        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("HyperParameters.fxml")));
//        center.setCenter(p);
    }
    public void evaluateFunction(ActionEvent event) throws IOException {
        getInstance().setBorderPane("Evaluater.fxml");

//        AnchorPane p = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("Evaluater.fxml")));
//        center.setCenter(p);
    }

}

package helper.home;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.layout.Pane;



public class Controller1 {

    private int layer_count = 0;

    @FXML
    private Label welcomeText;

    @FXML
    private Button layers;

    @FXML
    private Button add;

    @FXML
    private Button remove;

    @FXML
    private Button load_button;

    // Class information
    // addlayer();

    @FXML
    private Pane datasets_pane;

    @FXML
    private Pane layers_pane;

    @FXML
    private Pane optimization_pane;

    @FXML
    public void clearPanes() {
        datasets_pane.setVisible(false);
        layers_pane.setVisible(false);
        optimization_pane.setVisible(false);
    }

    @FXML
    public void configureDatasets() {
        clearPanes();
        datasets_pane.setVisible(true);
    }

    @FXML
    public void addLayer() {
        clearPanes();
        layers_pane.setVisible(true);
        //
    }

    @FXML
    public void showGraph() {
        clearPanes();
    }

    @FXML
    public void removeLayer() {
        -- layer_count;
        System.err.println("removeLayer : " + layer_count);
    }




    @FXML
    public void load_dataset() {
        // load dataset
        load_button.setVisible(false);


        // may be reset fileChooser ????
    }

    @FXML
    private CheckBox activation_checkbox;
    @FXML
    private CheckBox dense_checkbox;
    @FXML
    private CheckBox flatten_checkbox;
    @FXML
    private CheckBox reshape_checkbox;
    @FXML
    private CheckBox conv_2d_checkbox;
    @FXML
    private CheckBox depth_conv_2d_checkbox;

    @FXML
    private void clearCheckboxes () {
        activation_checkbox.setSelected(false);
        dense_checkbox.setSelected(false);
        flatten_checkbox.setSelected(false);
        reshape_checkbox.setSelected(false);
        conv_2d_checkbox.setSelected(false);
        depth_conv_2d_checkbox.setSelected(false);
    }

    @FXML
    private void checkBoxClicked () {
        clearCheckboxes();
    }

    private void getLayerConfig () {
        // get layer config from checkboxes

    }

    @FXML
    private void add_new_layer () {
        int type = 0;
        if (activation_checkbox.isSelected()) type = 0;
        if (dense_checkbox.isSelected()) type = 1;
        if (flatten_checkbox.isSelected()) type = 2;
        if (reshape_checkbox.isSelected()) type = 3;
        if (conv_2d_checkbox.isSelected()) type = 4;
        if (depth_conv_2d_checkbox.isSelected()) type = 5;
        LayerContainer.addLayer(type);
        layers_pane.setVisible(false);
    }


    @FXML
    private void printer () {
        System.err.println("lol");
    }
}






//    <?xml version="1.0" encoding="UTF-8"?>
//
//<?import javafx.scene.control.*?>
//<?import javafx.scene.layout.*?>
//
//<SplitPane dividerPositions="0.37299465240641705" maxHeight="-Infinity" maxWidth="-Infinity" minHeight="-Infinity" minWidth="-Infinity" prefHeight="400.0" prefWidth="600.0" xmlns="http://javafx.com/javafx/17.0.2-ea" xmlns:fx="http://javafx.com/fxml/1" fx:controller="com.helper.helper.home.shallow.helper.helper.home.shallow.HelloController">
//<items>
//<AnchorPane minHeight="0.0" minWidth="0.0" prefHeight="160.0" prefWidth="100.0">
//<children>
//<Button fx:id="layers" layoutX="46.0" layoutY="36.0" mnemonicParsing="false" onMouseClicked="#showPane" text="Layers" />
//</children></AnchorPane>
//<AnchorPane minHeight="0.0" minWidth="0.0" prefHeight="160.0" prefWidth="100.0">
//<children>
//<TitledPane fx:id="layer_cfg" animated="false" text="Layer config">
//<content>
//<AnchorPane minHeight="0.0" minWidth="0.0" prefHeight="180.0" prefWidth="200.0">
//<children>
//<Button fx:id="add" layoutX="30.0" layoutY="22.0" mnemonicParsing="false" onMouseClicked="#addLayer" text="add layer" />
//<Button fx:id="remove" layoutX="30.0" layoutY="78.0" mnemonicParsing="false" onMouseClicked="#removeLayer" text="remove layer" />
//</children>
//</AnchorPane>
//</content>
//</TitledPane>
//</children></AnchorPane>
//</items>
//</SplitPane>

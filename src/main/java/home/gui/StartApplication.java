package home.gui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class StartApplication extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(StartApplication.class.getResource("Main.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 925, 706);
        stage.setTitle("Shallow");
        stage.setScene(scene);

//        Connector connector = new Connector();
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}
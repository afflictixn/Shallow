package home.gui;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;

public class BasicController {
    @FXML
    private Label label;
    int i = 0;

    @FXML
    private Button button;

    public void Func(){
        ++i;
        label.setText("current value : " + i);
    }
}

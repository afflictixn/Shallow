package home;

import java.util.ArrayList;

public class LayerContainer {
    public static ArrayList<Layer> layers;

    LayerContainer () {
        layers = new ArrayList<>();
    }

    static void addLayer(int layer_type) {
        layers.add(new Layer(layer_type));
    }
}

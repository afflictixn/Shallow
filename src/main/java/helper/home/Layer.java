package helper.home;

public class Layer {
    private static int all_id = 0;

    private final int id;
    private int type;
    // 0 - activation
    // 1 - dense
    // 2 - flatten
    // 3 - reshape
    // 4 - 2d convolutional
    // 5 - depthwise 2d convolutional
    Layer (int type) {
        this.id = all_id++;
        this.type = type;
    }

    void changeType (int new_type) {
        this.type = new_type;
    }
}

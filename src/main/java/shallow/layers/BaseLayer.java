package shallow.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

public abstract class BaseLayer {
    Map<String, INDArray> cache;
    public BaseLayer(){
        cache = new HashMap<>();
    }
    public abstract INDArray forward(INDArray input);

    public abstract INDArray backward(INDArray derivatives);
}

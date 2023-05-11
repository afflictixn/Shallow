package shallow;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;
import shallow.layers.BaseLayer;
import shallow.layers.Linear;
import shallow.layers.ReLU;
import shallow.layers.configs.LinearLayerConfig;
import shallow.optimizers.Adam;
import shallow.utils.Utils;


import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
            Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
//        INDArray x = Nd4j.create(new float[]{1, 2, 3});
//        System.out.println(x.neg());
//        System.out.println(x);
//        System.out.println(x);
//        x.addi(Nd4j.create(new float[]{3, 4, 5}));
//        System.out.println(x);
//        System.out.println(Nd4j.ones(3).divi(x));
//        Sigmoid layer = new Sigmoid();
//        System.out.println(layer.forward(Nd4j.create(new double[]{10, 5, 2})));
        Linear layer = new Linear(new LinearLayerConfig(2, 3, "XavierNormal"));
        // m = 2
        INDArray X = Nd4j.create(new double [][] {{100}, {-10}, {-5}});
        BaseLayer act = new ReLU();
        System.out.println(act.forward(X));
        INDArray Y = Nd4j.create(new double [][] {{-200}, {-1}, {100}});
        System.out.println(act.backward(Y));
        System.out.println(X);
        System.out.println(Y);
//        INDArray res = X.mul(Y);
//        System.out.println(res);
//        BCELoss loss = new BCELoss();
//        System.out.println(loss.forward(X, Y).getDouble());
//        System.out.println(loss.backward());
        List<BaseLayer> layers = new ArrayList<>();
        Linear second = new Linear(new LinearLayerConfig(2, 2, "XavierNormal"));
        layers.add(second);
        Adam adam = new Adam(layers);
        adam.updateWeights(0.1, 1);
        INDArray temp = Nd4j.create(new double[]{1, 1e-10, 0.5, 0, -1});
        System.out.println(temp.dataType());
        INDArray clipped = Utils.get().clipByValue(temp, Utils.epsilon8, 1. - 1e-4);
        System.out.println("there");
    }
}
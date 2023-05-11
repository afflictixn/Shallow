package shallow.utils;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.ops.NDMath;

public class Utils {
    public static double epsilon4 = 1e-4;
    public static double epsilon8 = 1e-8;
    public static double epsilon12 = 1e-12;
    // clips in place
    public static INDArray clipi(INDArray X, double min_val, double max_val){
        INDArray min_mask = X.lt(min_val).castTo(DataType.FLOAT).muli(min_val);
        INDArray max_mask = X.gt(max_val).castTo(DataType.FLOAT).muli(max_val);
        X.muli(X.lt(max_val).castTo(DataType.FLOAT)).muli(X.gt(min_val).castTo(DataType.FLOAT)).addi(min_mask).addi(max_mask);
        return X;
    }
    // returns clipped copy in place
    public static INDArray clip(INDArray X, double min_val, double max_val){
        INDArray min_mask = X.lt(min_val).castTo(DataType.FLOAT).muli(min_val);
        INDArray max_mask = X.gt(max_val).castTo(DataType.FLOAT).muli(max_val);
        X.mul(X.lt(max_val).castTo(DataType.FLOAT)).muli(X.gt(min_val).castTo(DataType.FLOAT)).addi(min_mask).addi(max_mask);
        return X;
    }
    private static final NDMath math = new NDMath();
    public static NDMath get(){
        return math;
    }
}

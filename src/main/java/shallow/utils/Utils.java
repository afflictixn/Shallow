package shallow.utils;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.ops.NDMath;

import java.util.Random;
import java.util.stream.IntStream;

public class Utils {
    public static double epsilon4 = 1e-4;
    public static double epsilon8 = 1e-8;
    public static double epsilon12 = 1e-12;
    // clips in place
    public static INDArray clipi(INDArray X, double min_val, double max_val){
        INDArray minMask = X.lt(min_val).castTo(DataType.FLOAT).muli(min_val);
        INDArray maxMask = X.gt(max_val).castTo(DataType.FLOAT).muli(max_val);
        X.muli(X.lt(max_val).castTo(DataType.FLOAT)).muli(X.gt(min_val).castTo(DataType.FLOAT)).addi(minMask).addi(maxMask);
        return X;
    }
    // returns clipped copy in place
    public static INDArray clip(INDArray X, double min_val, double max_val){
        INDArray minMask = X.lt(min_val).castTo(DataType.FLOAT).muli(min_val);
        INDArray maxMask = X.gt(max_val).castTo(DataType.FLOAT).muli(max_val);
        X.mul(X.lt(max_val).castTo(DataType.FLOAT)).muli(X.gt(min_val).castTo(DataType.FLOAT)).addi(minMask).addi(maxMask);
        return X;
    }
    public static int[] randomPermutation(int n){
        int [] permutation = new int[n];
        IntStream.range(0, n).forEach(i -> permutation[i] = i);
        Random rand = new Random();
        for (int i = n - 1; i > 0; --i) {
            int j = rand.nextInt(i + 1);
            int temp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = temp;
        }
        return permutation;
    }
    private static final NDMath math = new NDMath();
    public static NDMath get(){
        return math;
    }
}

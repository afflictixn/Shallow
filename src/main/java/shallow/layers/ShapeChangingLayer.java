package shallow.layers;

public interface ShapeChangingLayer {
    // init can take data in 2 formats, either in format [batchSize, dim1, ...] or in format [dim1, ...],
    // i.e. in 2D convolutional layers if inShape is of length 4, it's considered to have batchSize in it
    // and if length is 3, it's considered to not have batchSize. The same hold for Linear layer and others.
    void init(long... inShape);
    long[] getOutputShape();
}

package shallow.layers.weight_init;

public enum WeightInitEnum {
    ZEROS,
    ONES,
    Normal,
    HeNormal,
    HeUniform,
    XavierNormal,
    XavierUniform;

    public WeightInitializer getWeightInitializer() throws UnsupportedOperationException {
        switch (this) {
            case ZEROS -> {
                return new ConstInit(0.0);
            }
            case ONES -> {
                return new ConstInit(1.0);
            }
            case Normal -> {
                return new NormalInit();
            }
            case HeNormal -> {
                return new HeNormalInit();
            }
            case HeUniform -> {
                return new HeUniformInit();
            }
            case XavierNormal -> {
                return new XavierNormalInit();
            }
            case XavierUniform -> {
                return new XavierUniformInit();
            }
        }
        throw new UnsupportedOperationException("Such weight initialization isn't implemented");
    }
}

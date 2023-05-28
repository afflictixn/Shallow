package shallow.layers.weight_init;

public enum WeightInitEnum {
    ZERO,
    HeNormal,
    HeUniform,
    XavierNormal,
    XavierUniform;

    public WeightInitializer getWeightInitializer() throws UnsupportedOperationException {
        switch (this) {
            case ZERO -> {
                return new ConstInit(0.0);
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

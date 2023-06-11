package shallow.losses;

public enum LossEnum {
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss;
    public BaseLoss getLoss(){
        switch (this){
            case CategoricalCrossEntropyLoss -> {
                return new CategoricalCrossEntropyLoss();
            }
            case BinaryCrossEntropyLoss -> {
                return new BinaryCrossEntropyLoss();
            }
            default -> {
                return null;
            }
        }
    }
}

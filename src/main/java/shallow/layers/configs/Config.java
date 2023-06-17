package shallow.layers.configs;

import shallow.layers.BaseLayer;

public interface Config {
    BaseLayer buildLayer();
    String getDescription();
}

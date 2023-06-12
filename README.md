# Shallow - Deep Learning Framework

Shallow is a deep learning framework originally established in Krakow, Poland.

## Dependencies

Almost all necessary dependencies for Shallow will be downloaded by Maven using pom.xml file.

Please ensure that there is no `module-info` file in your cloned repository, as the library Nd4j is not compatible with `module-info` files.

## JavaFX SDK

Make sure you have one of the newest versions of JavaFX SDK installed. The application works well with `javafx-sdk-20.0.1`, which can be downloaded, for instance, from [here](https://gluonhq.com/products/javafx/).

The JavaFX SDK library is used to import runtime dependencies of JavaFX, such as modules `javafx.controls`, `javafx.fxml`, and `javafx.web`, into the `StartApplication`. Follow the steps below to configure the run options:

1. Edit the run configuration of `StartApplication`.
2. Set the VM options to the following:

```bash
--module-path "${path_to_your_javafx_sdk}\javafx-sdk-20.0.1\lib" --add-modules javafx.controls,javafx.fxml,javafx.web

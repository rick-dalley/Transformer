# Rust Neural Network

This project is a simple implementation of a neural network written in **Rust**, inspired by the ideas presented in _"Make Your Own Neural Network"_ by Tariq Rashid. While the original book uses Python, this project aims to replicate the concepts in Rust as a learning exercise.

---

## Overview

- **Purpose**: This is a neural network implementation designed for educational purposes, not a robust production-level neural network library.
- **Training Data**: The neural network uses the [MNIST dataset](https://yann.lecun.com/exdb/mnist/) for training and testing. MNIST is a widely used dataset of handwritten digits (0–9).
- **Output**: After 10 training runs, the network outputs its confidence for each digit (0–9).

---

## Features

- Written in Rust for those interested in exploring neural network concepts in a lower-level language.
- Uses a simple feed-forward, backpropagation-based approach.
- Provides insights into how neural networks learn without relying on external libraries.

---

## Requirements

1. **MNIST Data**:

   - Download the MNIST dataset in CSV format from [Yann LeCun's MNIST page](https://yann.lecun.com/exdb/mnist/).
   - Place the files in a directory accessible to your project (e.g., `Data/mnist`).

2. **Rust Compiler**:

   - A modern Rust Compiler.

3. **Development Environment**:
   - Tested using VSCode and the terminal. The project should work in any Rust-compatible environment (e.g., the terminal or IDEs like IntelliJ IDEA, Visual Studio, etc.)

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rick-dalley/NeuralNetworkRust.git
   cd NeuralNetwork
   ```
2. Add the MNIST dataset:  
   • Place mnist_train.csv (training data) in the data directory.  
   • Create a config.json file in the same directory and ensure that you use this format:

   ```config.json
   {
      "data_file": "/path/to/your/data/mnist_train.csv",
      "label_column": 0,
      "input_nodes": 784,
      "hidden_nodes": 100,
      "scaling_factor": 255.0,
      "epochs": 1,
      "data_rows": 60000,
      "batch_size": 32,
      "output_classes": 10,
      "learning_rate": 0.3,
      "shuffle_data": true,
      "validation_split": 0.1
   }

   ```

   • Ensure that you update the <mark>data_file</mark> value to one that matches the location of your data file.  
   • Note that the rest of these settings assume that you are working with the mnist training data. If you are not, then you must update these settings with those appropriate for your data file.

3. Build and run the project:

   ```build & run
   cargo build
   cargo run
   ```

   For the best performance:

   ```release
   cargo build --release
   ./target/release/NeuralNetworkRust
   ```

4. Sample Output:  
    After running the application with the defaults and using the MNIST data set here is the output to the screen that was generated:

   ```Sample Output
       Model Configuration:
       Input Nodes: 784
       Hidden Nodes: 100
       Output Nodes: 10
       Epochs: 1
       Learning Rate: 0.3
       Scaling Factor: 255
       Shuffle Data: true
       Validation Split: 0.1
       Data Rows: 60000
       Split Index: 54000
       Data Location: /Users/richarddalley/Code/Rust/NeuralNetworkRust/data/mnist_train.csv
       Input-Hidden Weights Dimensions: 100x784
       Hidden-Output Weights Dimensions: 10x100
       Data Matrix Dimensions: 60000x784
       Shuffling data...
       Splitting data...
       Training data .......... .......... .......... .......... .......... ...

       Epoch 1/1 - Loss: 0.0932, Accuracy: 98.81%

       Training completed in 10.42s (hh:mm:ss.milliseconds)
   ```

   **Note**: The above output is an example. Your results may vary depending on:  
   • Random initialization of weights.  
   • Specific dataset splits (training vs. validation).  
   • System differences (e.g., floating-point precision, library versions).  
   • Adjustments to configuration parameters.

## Disclaimer

This project is a training exercise only and is not intended to be a comprehensive or robust implementation of a neural network. While you’re welcome to use this code as a starting point for your own learning, please note the following:  
 • It is not optimized for performance or large-scale use.  
 • It is not meant for production environments.  
 • There are alternatives to some modules which may offer optimizations, they have been commented out. You are free to try them to explore any changes they make to the performance, as a means of understanding some options for optimization.  
 • Contributions and feedback are welcome, but the project may lack features found in modern neural network libraries.

## Credits

    •    Book: “Make Your Own Neural Network” by Tariq Rashid.
    •    Dataset: MNIST.

## License

Feel free to use and modify this code for educational purposes. Attribution is appreciated but not required.

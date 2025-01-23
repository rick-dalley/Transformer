// module Model
// Richard Dalley

use crate::matrix::{Matrix, VectorType, Dot};
use crate::activation_functions::*;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use serde_json::from_reader;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Instant;
use std::io::{self, Write};

#[derive(Deserialize)]
pub struct Config {
    pub data_file: String,
    pub input_nodes: usize,
    pub hidden_nodes: usize,
    pub scaling_factor: f64,
    pub epochs: usize,
    pub data_rows: usize,
    pub output_classes: usize,
    pub learning_rate: f64,
    pub shuffle_data: bool,
    pub validation_split: f64,
    pub activation_functions: ActivationConfig, // Add this field
}

#[derive(Deserialize)]
pub struct ActivationConfig {
    pub hidden_layer: String,
    pub output_layer: String,
}
// Model
#[derive(Debug)] 
pub struct Model {
    input_nodes: usize,
    hidden_nodes: usize,
    output_nodes: usize,
    epochs: usize,
    learning_rate: f64,
    scaling_factor: f64,
    shuffle_data: bool,
    validation_split: f64,
    data_rows: usize,
    split_index: usize,
    data_location: String,

    input_hidden_weights: Matrix, // Assuming Matrix<f64>
    hidden_output_weights: Matrix,

    data: Matrix,
    training_data:Matrix,
    validation_data: Matrix,
    labels: Vec<usize>,
    training_labels: Vec<usize>,
    validation_labels: Vec<usize>,

    hidden_function: ActivationFunction, // Hidden layer activation
    output_function: ActivationFunction,
}


impl Model {

    // from_json - build a model from json
    pub fn from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the JSON file
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Parse the JSON into a Config struct
        let config: Config = from_reader(reader)?;

        // Initialize the Model struct
        let mut model = Self {
            input_nodes: config.input_nodes,
            hidden_nodes: config.hidden_nodes,
            output_nodes: config.output_classes,
            epochs: config.epochs,
            learning_rate: config.learning_rate,
            scaling_factor: config.scaling_factor,
            shuffle_data: config.shuffle_data,
            validation_split: config.validation_split,
            data_rows: config.data_rows,
            split_index: 0, // To be calculated during data processing
            data_location: config.data_file.clone(),
            labels: vec![],
            training_labels: vec![],
            validation_labels: vec![],
            input_hidden_weights: Matrix::zeros(config.hidden_nodes, config.input_nodes),
            hidden_output_weights: Matrix::zeros(config.output_classes, config.hidden_nodes),
            data: Matrix::zeros(config.data_rows, config.input_nodes),
            training_data: Matrix::zeros(0, config.input_nodes), // Placeholder until split
            validation_data: Matrix::zeros(0, config.input_nodes),
            hidden_function: get_activation_function(&config.activation_functions.hidden_layer, None),
            output_function: get_activation_function(&config.activation_functions.output_layer, None),
        };

    
        // Calculate split index based on the number of rows and validation split
        model.split_index = (config.data_rows as f64 * (1.0 - config.validation_split)) as usize;
        //assign weights to start
        model.input_hidden_weights.initialize_weights(config.input_nodes);
        model.hidden_output_weights.initialize_weights(config.hidden_nodes);

        Ok(model)
    }


    // load_data for training
    pub fn load_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .from_path(&self.data_location)?;

        // Reserve space for labels and data
        self.labels.reserve(self.data_rows);
        self.data = Matrix::zeros(self.data_rows, self.input_nodes);

        for (i, record) in reader.records().enumerate() {
            let record = record?;
            if i >= self.data_rows {
                break; // Stop reading after the specified number of rows
            }

            // First value is the label
            let label: usize = record[0].parse()?;
            self.labels.push(label);

            // Remaining values are the input data, normalized
            for (j, value) in record.iter().skip(1).enumerate() {
                let pixel: f64 = value.parse()?;
                self.data.data[i * self.input_nodes + j] = pixel / self.scaling_factor;
            }
        }

        // Call the shuffle and split functions
        if self.shuffle_data {
            self.shuffle_data();
        }

        if self.validation_split > 0.0 {
            self.split_data();
        } else {
            // No split, assign all data to training
            self.split_index = self.data_rows;
            self.training_data = self.data.clone();
            self.training_labels = self.labels.clone();
        }

        Ok(())
    }

    pub fn shuffle_data(&mut self) {
        // Placeholder for now
        println!("Shuffling data...");

        let mut indices: Vec<usize> = (0..self.data_rows).collect(); // Create indices
        let mut rng = thread_rng();
        indices.shuffle(&mut rng); // Shuffle indices

        // Reorder data and labels
        let mut shuffled_data = Matrix::zeros(self.data.rows, self.data.cols);
        let mut shuffled_labels = Vec::with_capacity(self.labels.len());

        for (new_idx, &original_idx) in indices.iter().enumerate() {
            // Copy row by row from the original data matrix
            for col in 0..self.data.cols {
                shuffled_data.data[new_idx * self.data.cols + col] =
                    self.data.data[original_idx * self.data.cols + col];
            }
            // Copy corresponding label
            shuffled_labels.push(self.labels[original_idx]);
        }

        self.data = shuffled_data; // Update the data matrix
        self.labels = shuffled_labels; // Update the labels vector
    }

    pub fn split_data(&mut self) {
        println!("Splitting data...");
           
        // Split data into training and validation sets
        self.training_data = Matrix::zeros(self.split_index, self.data.cols);
        self.validation_data = Matrix::zeros(self.data.rows - self.split_index, self.data.cols);

        for i in 0..self.split_index {
            for j in 0..self.data.cols {
                self.training_data.data[i * self.data.cols + j] = self.data.data[i * self.data.cols + j];
            }
        }

        for i in self.split_index..self.data.rows {
            for j in 0..self.data.cols {
                let validation_row = i - self.split_index;
                self.validation_data.data[validation_row * self.data.cols + j] =
                    self.data.data[i * self.data.cols + j];
            }
        }

        // Split labels into training and validation sets
        self.training_labels = self.labels[..self.split_index].to_vec();
        self.validation_labels = self.labels[self.split_index..].to_vec();
    }

    pub fn train(&mut self, show_progress: bool) {

        let start_time = Instant::now(); // Capture the start time
        
        let (hidden_activation, output_activation) = self.resolve_activation_functions();
        
        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;


            for i in 0..self.training_data.rows {
                let input_row = self.training_data.row_slice(i).expect("Row out of bounds");

                // Convert the row slice into a column vector
                let input = Matrix::from_vector(input_row.to_vec(), VectorType::Column);                
                
                let mut target = Matrix::zeros(self.output_nodes, 1);
                target.data[self.training_labels[i]] = 0.99;

                // Train the layer
                self.train_layer(&input, &target, &hidden_activation,&output_activation);

                // Forward pass for metrics
                let (_, final_outputs) = self.forward_pass(&input, &hidden_activation, &output_activation);
                let loss = self.calculate_loss(&final_outputs, self.training_labels[i]);
                total_loss += loss;

                // Accuracy calculation
                let predicted_label = final_outputs.argmax(); // Find max index
                if predicted_label == self.training_labels[i] {
                    correct_predictions += 1;
                }

                // Show progress
                if i > 0 {
                    if show_progress && i % 1000 == 0 {
                        print!(".");
                        io::stdout().flush().unwrap();
                        if i % 10000 == 0 {
                           print!(" ");
                        }
                    }
                } else {
                    print!(" Training data ");
                }
            }

            //start a new line for the next epoch    
            println!();

            // Metrics for the epoch
            if show_progress {
                let average_loss = total_loss / self.training_data.rows as f64;
                let accuracy = (correct_predictions as f64 / self.training_data.rows as f64) * 100.0;
                println!(
                    "\nEpoch {}/{} - Loss: {:.4}, Accuracy: {:.2}%",
                    epoch + 1,
                    self.epochs,
                    average_loss,
                    accuracy
                );
            }
        }
            // Calculate and report elapsed time
        let elapsed_time = start_time.elapsed();
        println!(
            "\nTraining completed in {:.2?} (hh:mm:ss.milliseconds)",
            elapsed_time
        );

    }


    // forward_pass with activation functions
fn forward_pass (
    &self,
    input: &Matrix,
    hidden_activation: &Box<dyn Fn(f64) -> f64>,
    output_activation: &Box<dyn Fn(f64) -> f64>,
) -> (Matrix, Matrix)
{
    // Hidden layer
    let hidden_inputs = self.input_hidden_weights.dot(input);
    let hidden_outputs = hidden_inputs.apply(hidden_activation);

    // Output layer
    let final_inputs = self.hidden_output_weights.dot(&hidden_outputs);
    let final_outputs = final_inputs.apply(output_activation);

    (hidden_outputs, final_outputs)
}
    fn calculate_loss(
        &self, output: &Matrix, 
        true_label: usize) -> f64 {
        let mut loss = 0.0;
        for i in 0..self.output_nodes {
            let predicted = output.data[i];
            let target = if i == true_label { 1.0 } else { 0.0 };
            loss -= target * (predicted + 1e-7).ln(); // Avoid log(0)
        }
        loss
    }

    fn train_layer(
        &mut self, 
        input: &Matrix, 
        target: &Matrix,
        hidden_activation: &Box<dyn Fn(f64) -> f64>,
        output_activation: &Box<dyn Fn(f64) -> f64>,
    ) {
        // Forward pass
        let (hidden_outputs, final_outputs) = self.forward_pass(input, hidden_activation, output_activation);

        // Calculate errors
        let output_errors = target - &final_outputs;
        let hidden_errors = self.hidden_output_weights.transpose().dot(&output_errors);

        // Gradients for output layer
        let output_gradients = final_outputs.apply(|x| x * (1.0 - x));
        let weight_delta_output = (output_errors * &output_gradients)
            .dot(&hidden_outputs.transpose())
            * self.learning_rate;

        self.hidden_output_weights += weight_delta_output;

        // Gradients for hidden layer
        let hidden_gradients = hidden_outputs.apply(|x| x * (1.0 - x));
        let weight_delta_input = (hidden_errors * &hidden_gradients)
            .dot(&input.transpose())
            * self.learning_rate;

        self.input_hidden_weights += weight_delta_input;
    }
    
    // determine the activation functions
    fn resolve_activation_functions(&self) -> (Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>) {
        let hidden_activation: Box<dyn Fn(f64) -> f64> = match self.hidden_function {
            ActivationFunction::Sigmoid => Box::new(sigmoid),
            ActivationFunction::ReLU => Box::new(relu),
            ActivationFunction::Tanh => Box::new(tanh),
            ActivationFunction::Swish => Box::new(swish),
            _ => panic!("Unsupported hidden activation function"),
        };

        let output_activation: Box<dyn Fn(f64) -> f64> = match self.output_function {
            ActivationFunction::Sigmoid => Box::new(sigmoid),
            ActivationFunction::Swish => Box::new(swish),
            ActivationFunction::Softmax => panic!("Softmax should be applied to vectors"),
            _ => panic!("Unsupported output activation function"),
        };

        (hidden_activation, output_activation)
    }

    // print the config
    pub fn print_config(&self) {
        println!("Model Configuration:");
        println!("  Input Nodes: {}", self.input_nodes);
        println!("  Hidden Nodes: {}", self.hidden_nodes);
        println!("  Output Nodes: {}", self.output_nodes);
        println!("  Epochs: {}", self.epochs);
        println!("  Learning Rate: {}", self.learning_rate);
        println!("  Scaling Factor: {}", self.scaling_factor);
        println!("  Shuffle Data: {}", self.shuffle_data);
        println!("  Validation Split: {}", self.validation_split);
        println!("  Data Rows: {}", self.data_rows);
        println!("  Split Index: {}", self.split_index);
        println!("  Data Location: {}", self.data_location);
        println!("  Input-Hidden Weights Dimensions: {}x{}", self.input_hidden_weights.rows, self.input_hidden_weights.cols);
        println!("  Hidden-Output Weights Dimensions: {}x{}", self.hidden_output_weights.rows, self.hidden_output_weights.cols);
        println!("  Data Matrix Dimensions: {}x{}", self.data.rows, self.data.cols);
        println!("  Hidden Layer Activation Function: {}", self.hidden_function);
        println!("  Output Layer Activation Function: {}", self.output_function);
        println!("  Data Matrix Dimensions: {}x{}", self.data.rows, self.data.cols);

    }
}

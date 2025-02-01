// module Model
// Richard Dalley

// uses
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use crate::activation_functions::{self};
use crate::config;
use crate::matrix::{Matrix, Dot};
use crate::data_loader::DataLoader;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{ BufWriter, Write};
use std::fs::OpenOptions;
use plotters::prelude::*;
const DATA_PATH: &str = "./data/{1}/{2}";


// enums
pub enum TaskEnum {
    Classification(ClassificationTaskImpl),
    Regression(RegressionTaskImpl),
}

// Traits
pub trait TaskTrait {
    fn transformer_layer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait;

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait;

    fn backward_transformer<T>(&self, model: &Model<T>, outputs: &Matrix, predictions: &Matrix, output_errors: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait;

    fn backward_feedforward<T>(&self, model: &Model<T>, gradients: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait;

    fn backward_multi_head_attention<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix,
        predictions: &Matrix,
    ) -> Matrix
    where
        T: TaskTrait + TrainTrait;

    fn update_weights<T>(&self, model: &mut Model<T>, gradients: &Matrix, learning_rate: f64)
    where
        T: TaskTrait + TrainTrait;
    
    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait;
}

pub trait TrainTrait {
    fn compute_loss(&self, outputs: &Matrix, targets: &Matrix) -> f64;
    fn compute_output_errors(&self, outputs: &Matrix, targets: &Matrix) -> Matrix;
    fn compute_accuracy(&self, outputs: &Matrix, labels: &[usize]) -> f64;
    fn compute_final_output<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait; // Ensure Model<T> supports both traits

}

// Model
pub struct Model<'a, T: TaskTrait + TrainTrait> {
    pub data_loader: &'a mut DataLoader,
    epochs: usize,
    checkpoint: usize,
    learning_rate: f64,
    batch_size: usize,
    num_layers: usize,
    num_heads: usize,
    embed_dim: usize,
    num_classes: usize,
    output_attention_weights: Vec<Matrix>,
    ff_hidden_weights: Matrix,
    ff_output_weights: Matrix,
    final_output_weights: Matrix,
    embedding_matrix: Matrix,
    learning_task: config::LearningTask,
    activation_fn: Box<dyn activation_functions::ActivationTrait>,
    derivative_fn: Box<dyn activation_functions::ActivationTrait>,
    project: String,
    task: T, // Task-specific behavior
}

#[derive(Serialize, Deserialize)]
struct ModelCheckpoint {
    final_output_weights: Vec<f64>,
    ff_hidden_weights: Vec<f64>,
    ff_output_weights: Vec<f64>,
}

impl<'a, T: TaskTrait + TrainTrait> Model<'a, T> {

    pub fn from_json(
        config: &config::Config,
        data_loader: &'a mut DataLoader,
        project_name: &str,
        task: T, // Pass the task-specific implementation
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the JSON file
        let project = DATA_PATH.replace("{1}", project_name);
        let config_clone = config.clone();
        let num_classes = config.num_classes;
        let epochs = config.epochs;
        let checkpoint = config.checkpoint_interval;
        let learning_rate = config.learning_rate;
        let learning_task = config.learning_task;
        let num_heads = config.num_heads;
        let num_layers = config.num_layers;
        let batch_size = config.batch_size;
        let embed_dim = config.model_dimensions;
        let output_attention_weights = (0..config.num_heads)
            .map(|_| Matrix::random(config.model_dimensions / config.num_heads, config.model_dimensions / config.num_heads))
            .collect();

        let embedding_matrix = Matrix::random(config.vocab_size, config.model_dimensions);

        let (activation_fn, derivative_fn) = activation_functions::get_activation_and_derivative(&config_clone);


        let ff_hidden_weights = if learning_task == config::LearningTask::Classification {
            Matrix::random(config.model_dimensions, config.num_classes) // Fix for classification
        } else {
            Matrix::random(config.model_dimensions, config.hidden_dimensions) // Fix for regression
        };


        let ff_output_weights = if learning_task == config::LearningTask::Classification {
            Matrix::random(config.hidden_dimensions, num_classes) // Fix for classification
        } else {
            Matrix::random(config.hidden_dimensions, 1) // Fix for regression
        };

        let final_output_weights = if learning_task == config::LearningTask::Classification {
            // For classification, the number of classes defines the output size
            Matrix::random(config.model_dimensions, num_classes)
        } else {
            // For regression, the output size is always 1
            Matrix::random(config.model_dimensions, 1)
        };

        // Initialize the Model struct
        let model = Self {
            data_loader,
            epochs,
            checkpoint,
            learning_rate,
            learning_task,
            batch_size,
            num_heads,
            num_layers,
            embed_dim,
            num_classes,
            output_attention_weights,
            ff_hidden_weights,
            ff_output_weights,
            final_output_weights,
            embedding_matrix,
            activation_fn,
            derivative_fn,
            project,
            task, // Initialize task-specific behavior
        };

        Ok(model)
    }

    // Use activation functions in methods
    pub fn apply_activation_fn(&self, x: f64) -> f64 {
        self.activation_fn.apply(x)
    }

    // Computes query (Q), key (K), and value (V) matrices, and applies the attention formula:
    // Attention}(Q, K, V) = softmax(QK^T / √d_k) * V
    fn scaled_dot_product_attention(
        &self,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
    ) -> Matrix {
        let d_k = query.cols as f64;
        // Compute the attention scores
        let scores = query.dot(&key.transpose()) / d_k.sqrt();

        // Apply softmax row-wise to the attention scores
        let attention_weights = scores.softmax(); // Calls the Matrix's softmax method on scores

        // Multiply the attention weights by the value matrix
        attention_weights.dot(value)
    }

    pub fn embedding(&self, input: &Vec<usize>) -> Matrix {
        // Assume `embedding_matrix` is precomputed and stored in `self.embedding_matrix`
        let embedding_matrix = &self.embedding_matrix;
        let embed_dim = embedding_matrix.cols; // Embedding dimension derived from matrix

        // Create the result matrix
        let mut embeddings = Matrix::zeros(input.len(), embed_dim);

        // Use iterators and batch copy
        embeddings
            .data
            .chunks_mut(embed_dim) // Split result matrix into mutable row chunks
            .zip(input.iter()) // Pair each row with its corresponding token index
            .for_each(|(row, &token)| {
                let start = token * embed_dim;
                let end = start + embed_dim;
                row.copy_from_slice(&embedding_matrix.data[start..end]); // Copy the entire row
            });

        embeddings
    }

    pub fn positional_encoding(&self, embed_dim: usize) -> Matrix {
        // Create a single row of positional encoding
        let mut encoding = Matrix::zeros(1, embed_dim); // One row for broadcasting
        for i in 0..embed_dim {
            if i % 2 == 0 {
                encoding.data[i] = (0.0 / 10000f64.powf(i as f64 / embed_dim as f64)).sin();
            } else {
                encoding.data[i] = (0.0 / 10000f64.powf(i as f64 / embed_dim as f64)).cos();
            }
        }
        encoding
    }

    pub fn multi_head_attention(
        &self,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
        num_heads: usize,
        embed_dim: usize,
    ) -> Matrix {
        // Split into multiple heads
        let head_dim = embed_dim / num_heads;
        let mut attention_heads = Vec::new();

        for head in 0..num_heads {
            // Extract the per-head matrices
            let q = query.extract_head(head, head_dim);
            let k = key.extract_head(head, head_dim);
            let v = value.extract_head(head, head_dim);

            // Perform scaled dot-product attention
            let scaled_attention = self.scaled_dot_product_attention(&q, &k, &v);

            // Use the per-head output attention weights
            let transformed = scaled_attention.dot(&self.output_attention_weights[head]);

            attention_heads.push(transformed);
        }

        // Concatenate heads and return the combined result
        Matrix::concat_heads(&attention_heads)
    }

    pub fn layer_norm(&self, input: &Matrix) -> Matrix {
        let epsilon = 1e-6;
        let mean = input.mean();
        let variance = input.variance();
        input.apply(|x| (x - mean) / (variance + epsilon).sqrt())
    }


    // pub fn update_weights(&mut self, gradients: &Matrix, learning_rate: f64) {
    //     // Aggregate gradients across the batch

    //     let aggregated_gradients = gradients.mean_axis(0); // (1, 512)

    //     if self.learning_task == config::LearningTask::Classification {
    //         let expanded_gradients = aggregated_gradients.broadcast(self.final_output_weights.cols).transpose(); // (512, num_classes)
    //         self.final_output_weights -= expanded_gradients * learning_rate;
    //     } else {
    //         let transposed_gradients = aggregated_gradients.transpose(); // (512, 1)
    //         self.final_output_weights -= transposed_gradients * learning_rate;
    //     }
    // }
pub fn update_weights(&mut self, gradients: &Matrix, learning_rate: f64) {


    // Define gradient clipping threshold
    let clip_threshold = 1.0;

    // Clip gradients to prevent explosion while maintaining shape
    let clipped_gradients = gradients.apply(|g| g.max(-clip_threshold).min(clip_threshold));


    // Aggregate gradients
    let aggregated_gradients = clipped_gradients.mean_axis(0); 

    if self.learning_task == config::LearningTask::Classification {
        // Ensure shape matches (512, num_classes)
        let expanded_gradients = aggregated_gradients.broadcast(self.final_output_weights.cols);
        

        // Fix: Transpose before subtraction if needed
        if expanded_gradients.rows != self.final_output_weights.rows {
            self.final_output_weights -= expanded_gradients.transpose() * learning_rate;
        } else {
            self.final_output_weights -= expanded_gradients * learning_rate;
        }
    } else {
        // Regression: Ensure proper shape (512, 1)
        let transposed_gradients = aggregated_gradients.transpose();


        self.final_output_weights -= transposed_gradients * learning_rate;
    }

}

    pub fn train(&mut self) {
        // Track training time
        let start_time = Instant::now();
        let mut loss_history: Vec<f64> = Vec::new();
        let mut accuracy_history: Vec<f64> = Vec::new();
        let checkpoint_location = self.project.replace("{2}", "model_checkpoint.json");
        let lossplot_location = self.project.replace("{2}", "loss_plot.png");

        // Progress bar setup
        let iterations: u64 = (self.epochs * (self.data_loader.training_data.rows / self.batch_size)) as u64;
        let pb = ProgressBar::new(iterations);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            let mut total_samples = 0;
            // Shuffle training data and labels
            DataLoader::shuffle_data(&mut self.data_loader.training_data, &mut self.data_loader.training_labels);

            for i in (0..self.data_loader.training_data.rows).step_by(self.batch_size) {
                // Prepare a batch of data
                let batch_data = self.data_loader.training_data.slice(i, i + self.batch_size);
                let batch_labels = &self.data_loader.training_labels[i..(i + self.batch_size).min(self.data_loader.training_labels.len())];

                // Forward pass
                let predictions = self.task.forward_transformer(self, &batch_data);
                let outputs = self.task.compute_final_output(self, &predictions);

                let target_batch = if self.learning_task == config::LearningTask::Classification {
                    Matrix::from_labels(batch_labels, self.num_classes) // One-hot encode labels
                } else {
                    Matrix::new(batch_labels.len(), 1, batch_labels.iter().map(|&x| x as f64).collect()) // Keep numeric values for regression
                };

                let batch_loss = self.task.compute_loss(&outputs, &target_batch);
                total_loss += batch_loss;

                // Compute accuracy using the TrainTrait
                let accuracy = self.task.compute_accuracy(&outputs, batch_labels);
                correct_predictions += (accuracy / 100.0 * batch_labels.len() as f64) as usize;
                total_samples += batch_labels.len();

                // Backward pass
                let output_errors = self.task.compute_output_errors(&outputs, &target_batch);

                let expanded_output_errors = if self.learning_task == config::LearningTask::Classification {
                    output_errors.clone() 
                } else {
                    output_errors.repeat_columns(self.embed_dim) 
                };

                let predictions_clone = predictions.clone();
                if self.learning_task == config::LearningTask::Classification {
                    // Handle classification-specific logic
                    let softmax_outputs = outputs.softmax(); // Ensure predictions are probabilities
                    let mut classification_errors = self.task.backward_transformer(self, &softmax_outputs, &predictions_clone, &expanded_output_errors);
                    classification_errors /= self.batch_size as f64;
                    self.update_weights(&classification_errors, self.learning_rate);

                } else {
                    // Handle regression logic (unchanged)
                    let attention_errors = self.task.backward_transformer(self, &outputs, &predictions_clone, &expanded_output_errors);
                    self.update_weights(&attention_errors, self.learning_rate);
                }

                // Update progress bar
                pb.inc(1);
            }

            // Check if it's time to do a checkpoint
            if epoch % self.checkpoint == 0 {
                println!("Saving...");
                self.save_checkpoint(checkpoint_location.as_str()).expect("Failed to save model.");
            }

            let avg_loss = total_loss / self.data_loader.training_data.rows as f64;
            let accuracy = (correct_predictions as f64 / total_samples as f64) * 100.0;

            loss_history.push(avg_loss);
            accuracy_history.push(accuracy);

            // Log the loss for this epoch
            self.log_training_metrics(epoch, avg_loss, accuracy, &self.project);

            println!("Epoch {}/{} - Loss: {:.4}", epoch + 1, self.epochs, avg_loss);
        }

        self.plot_loss_curve(loss_history, lossplot_location.as_str())
            .expect("Failed to generate loss plot");

        let elapsed_time = start_time.elapsed();
        println!(
            "\nTraining completed in {:.2?} (hh:mm:ss.milliseconds)",
            elapsed_time
        );
    }
    
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let checkpoint = ModelCheckpoint {
            final_output_weights: self.final_output_weights.data.clone(),
            ff_hidden_weights: self.ff_hidden_weights.data.clone(),
            ff_output_weights: self.ff_output_weights.data.clone(),
        };

        let json = serde_json::to_string(&checkpoint)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn print_config(&self) {
        println!("Model Configuration:");
        println!("  Epochs: {}", self.epochs);
        println!("  Learning Rate: {:.5}", self.learning_rate);
        println!("  Batch Size: {}", self.batch_size);
        println!("  Sequence Length: {}", self.data_loader.sequence_length);
        println!("  Validation Split: {:.2}%", self.data_loader.validation_split * 100.0);
        println!("  Split Index: {}", self.data_loader.split_index);
        println!("  Number of Layers: {}", self.num_layers);
        println!("  Number of Heads: {}", self.num_heads);
        println!("  Embedding Dimension: {}", self.embed_dim);
        println!("  Data Location: {}", self.data_loader.data_location);

        // Data dimensions
        println!("  Training Data: {} rows x {} cols", self.data_loader.training_data.rows, self.data_loader.training_data.cols);
        if self.data_loader.cap_data_rows {
            println!("  Capped for debugging for this run");
        }

        println!("  Validation Data: {} rows x {} cols", self.data_loader.validation_data.rows, self.data_loader.validation_data.cols);
        // Labels info
        println!("  Training Labels: {} total", self.data_loader.training_labels.len());
        println!("  Validation Labels: {} total", self.data_loader.validation_labels.len());

        // Check attention weights
        println!("  Attention Weights: {} heads", self.output_attention_weights.len());
        if let Some(first_attention_weight) = self.output_attention_weights.first() {
            println!(
                "  Per-Head Attention Weights: {} rows x {} cols",
                first_attention_weight.rows, first_attention_weight.cols
            );
        }

        // Feedforward weights
        println!(
            "  Feedforward Hidden Weights: {} rows x {} cols",
            self.ff_hidden_weights.rows, self.ff_hidden_weights.cols
        );
        println!(
            "  Feedforward Output Weights: {} rows x {} cols",
            self.ff_output_weights.rows, self.ff_output_weights.cols
        );

        // Final output weights
        println!(
            "  Final Output Weights: {} rows x {} cols",
            self.final_output_weights.rows, self.final_output_weights.cols
        );
    }

    // log training metrics
    pub fn log_training_metrics(&self, epoch: usize, loss: f64, accuracy: f64, project: &str) {
        let training_log_location = project.replace("{2}", "training_log.csv");
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(training_log_location)
            .expect("Failed to open log file");

        writeln!(file, "{},{},{}", epoch, loss, accuracy).expect("Failed to write log.");
    }

    // plot loss curve
    pub fn plot_loss_curve(&self, loss_values: Vec<f64>, location: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(&location, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Loss Curve", ("sans-serif", 50))
            .build_cartesian_2d(0..loss_values.len(), 0.0..loss_values.iter().cloned().fold(f64::INFINITY, f64::min))?;

        chart.draw_series(LineSeries::new(
            loss_values.iter().enumerate().map(|(i, &loss)| (i, loss)),
            &BLUE,
        ))?;

        Ok(())
    }

    // predict
    pub fn predict(&self, input: &Matrix) -> Matrix {
        let transformed = self.task.forward_transformer(self, input);
        self.task.compute_final_output(self, &transformed)
    }

    pub fn evaluate(&self, filename: Option<&str>) {
        let predictions = self.predict(&self.data_loader.validation_data);

        if let Some(file) = filename {
            let file = File::create(file).expect("Failed to create prediction file");
            let mut writer = BufWriter::new(file);

            for (i, pred) in predictions.data.iter().enumerate() {
                writeln!(writer, "Sample {}: {:?}", i + 1, pred).expect("Failed to write to file");
            }

            println!("Predictions saved to {:?}", filename);
        } else {
            println!("Predictions:");
            for (i, pred) in predictions.data.iter().enumerate() {
                println!("Sample {}: {:?}", i + 1, pred);
            }
        }
    }

    // Delegate to the task-specific implementation
}

impl TaskTrait for TaskEnum {
    fn transformer_layer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.transformer_layer(model, input),
            TaskEnum::Regression(task) => task.transformer_layer(model, input),
        }
    }

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.forward_transformer(model, input),
            TaskEnum::Regression(task) => task.forward_transformer(model, input),
        }
    }

    fn backward_transformer<T>(&self, model: &Model<T>, outputs: &Matrix, predictions: &Matrix, output_errors: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.backward_transformer(model, outputs, predictions, output_errors),
            TaskEnum::Regression(task) => task.backward_transformer(model,outputs, predictions, output_errors),
        }
    }

    fn backward_feedforward<T>(&self, model: &Model<T>, gradients: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.backward_feedforward(model, gradients),
            TaskEnum::Regression(task) => task.backward_feedforward(model, gradients),
        }
    }

    fn backward_multi_head_attention<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix,
        predictions: &Matrix,
    ) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.backward_multi_head_attention(model, gradients, predictions),
            TaskEnum::Regression(task) => task.backward_multi_head_attention(model, gradients, predictions),
        }
    }

    fn update_weights<T>(&self, model: &mut Model<T>, gradients: &Matrix, learning_rate: f64)
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.update_weights(model, gradients, learning_rate),
            TaskEnum::Regression(task) => task.update_weights(model, gradients, learning_rate),
        }
    }

    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.feedforward_network(model, input),
            TaskEnum::Regression(task) => task.feedforward_network(model, input),
        }
    }

}


// Implement TaskTrait for ClassificationTask and RegressionTask
pub struct ClassificationTaskImpl;

impl TaskTrait for ClassificationTaskImpl {

    fn transformer_layer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait, 
    {
        let attention_output = model.multi_head_attention(input, input, input, model.num_heads, model.embed_dim);
        let residual_sum = input + &attention_output; 
        let attention_residual = model.layer_norm(&residual_sum); 
        
        let ff_output = model.task.feedforward_network(model, &attention_residual);
        let ff_residual = attention_residual + ff_output; // Ensure residual connection
        
        model.layer_norm(&ff_residual) 
    }

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let token_indices = input.column_to_indices(0);
        let mut x = model.embedding(&token_indices);

        let positional_enc = model.positional_encoding(model.embed_dim);
        x = x.add_broadcast(&positional_enc);

        for _ in 0..model.num_layers {
            x = self.transformer_layer(model, &x);
        }
        x
    }

    fn backward_transformer<T>(&self, model: &Model<T>, outputs: &Matrix, predictions: &Matrix, output_errors: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        // Compute the gradient of the softmax output
        let softmax_gradients = outputs.softmax_gradient(); // Derivative of softmax

        // Multiply by the output errors to get the gradients
        let gradients = &softmax_gradients * output_errors;

        // Backpropagate through the feedforward network
        let ff_gradients = self.backward_feedforward(model, &gradients);


        // Backpropagate through the multi-head attention
        let attention_gradients = self.backward_multi_head_attention(model, &ff_gradients, predictions);
        
        // Return the final gradients
        attention_gradients
    }

    fn backward_feedforward<T>(&self, model: &Model<T>, gradients: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {

        let transposed = &model.ff_hidden_weights.transpose();

        // Derivatives through the second linear layer
        let grad_ff_output_weights = gradients.dot(transposed);

        // Backpropagate activation function using the stored activation derivative function
        let grad_hidden = grad_ff_output_weights.apply(|x| model.derivative_fn.apply(x));

        grad_hidden
    }

    fn backward_multi_head_attention<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix,
        predictions: &Matrix,
    ) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let head_dim = model.embed_dim / model.num_heads;
        let mut attention_gradients = Matrix::zeros(gradients.rows, model.embed_dim);

        for head in 0..model.num_heads {
            // Extract gradients for this head
            let grad_attention = gradients.extract_head(head, head_dim);

            // Compute gradients for queries, keys, and values
            let grad_query = grad_attention.dot(&model.output_attention_weights[head].transpose());
            let pred_head = predictions.extract_head(head, head_dim);
            let grad_key = grad_attention.transpose().dot(&pred_head);
            // Map grad_key back to embedding space
            let grad_key_reduced = pred_head.dot(&grad_key.transpose());

            // Compute value gradients
            let grad_value = grad_attention.dot(&model.output_attention_weights[head]);

            // Accumulate gradients for queries, keys, and values
            attention_gradients.add_head(&grad_query, head, head_dim);
            attention_gradients.add_head(&grad_key_reduced, head, head_dim);
            attention_gradients.add_head(&grad_value, head, head_dim);
        }

        attention_gradients
    }

    fn update_weights<T>(&self, model: &mut Model<T>, gradients: &Matrix, learning_rate: f64)
    where
        T: TaskTrait + TrainTrait, // Add TrainTrait bound
    {
        // Update the final output weights for classification
        let output_gradients = gradients.dot(&model.final_output_weights.transpose());
        model.final_output_weights -= &output_gradients * learning_rate;

        // Update the feedforward network weights
        let ff_output_gradients = gradients.dot(&model.ff_output_weights.transpose());
        model.ff_output_weights -= &ff_output_gradients * learning_rate;

        let ff_hidden_gradients = gradients.dot(&model.ff_hidden_weights.transpose());
        model.ff_hidden_weights -= &ff_hidden_gradients * learning_rate;
    }

    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let hidden = input.dot(&model.ff_hidden_weights) // (32, 512) ⋅ (512, 2048) → (32, 2048)
            .apply(|x| model.apply_activation_fn(x));

        hidden.dot(&model.ff_hidden_weights.transpose()) // (32, 2048) ⋅ (2048, 512) → (32, 512)
    }

}


impl TrainTrait for TaskEnum {
    fn compute_loss(&self, outputs: &Matrix, targets: &Matrix) -> f64 {
        match self {
            TaskEnum::Classification(task) => task.compute_loss(outputs, targets),
            TaskEnum::Regression(task) => task.compute_loss(outputs, targets),
        }
    }

    fn compute_output_errors(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        match self {
            TaskEnum::Classification(task) => task.compute_output_errors(outputs, targets),
            TaskEnum::Regression(task) => task.compute_output_errors(outputs, targets),
        }
    }

    fn compute_accuracy(&self, outputs: &Matrix, labels: &[usize]) -> f64 {
        match self {
            TaskEnum::Classification(task) => task.compute_accuracy(outputs, labels),
            TaskEnum::Regression(task) => task.compute_accuracy(outputs, labels),
        }
    }

    fn compute_final_output<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.compute_final_output(model, input),
            TaskEnum::Regression(task) => task.compute_final_output(model, input),
        }
    }


 }


pub struct RegressionTaskImpl;

impl TaskTrait for RegressionTaskImpl {
    fn transformer_layer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let attention_output = model.multi_head_attention(input, input, input, model.num_heads, model.embed_dim);
        let residual_sum = input + &attention_output;
        let attention_residual = model.layer_norm(&residual_sum);
        let ff_output = model.task.feedforward_network(model, &attention_residual);
        ff_output.repeat_columns(model.embed_dim)
    }

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let token_indices = input.column_to_indices(0);
        let mut x = model.embedding(&token_indices);
        let positional_enc = model.positional_encoding(model.embed_dim);
        x = x.add_broadcast(&positional_enc);

        for _ in 0..model.num_layers {
            x = self.transformer_layer(model, &x);
        }

        x
    }

   fn backward_transformer<T>(&self, model: &Model<T>, _outputs: &Matrix, predictions: &Matrix, output_errors: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        // For regression, the output errors are the same as the gradients
        let gradients = output_errors.clone();

        // Backpropagate through the feedforward network
        let ff_gradients = self.backward_feedforward(model, &gradients);

        // Backpropagate through the multi-head attention
        let attention_gradients = self.backward_multi_head_attention(model, &ff_gradients, predictions);

        // Return the final gradients
        attention_gradients
    }


   fn backward_feedforward<T>(&self, model: &Model<T>, gradients: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        // Derivatives through the second linear layer
        let grad_ff_output_weights = gradients.dot(&model.ff_hidden_weights);

        // Backpropagate activation function using the stored activation derivative function
        let grad_hidden = grad_ff_output_weights.apply(|x| model.derivative_fn.apply(x));

        grad_hidden.dot(&model.ff_hidden_weights.transpose())
    }

    fn backward_multi_head_attention<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix,
        predictions: &Matrix,
    ) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let head_dim = model.embed_dim / model.num_heads;
        let mut attention_gradients = Matrix::zeros(gradients.rows, model.embed_dim);

        for head in 0..model.num_heads {
            // Extract gradients for this head
            let grad_attention = gradients.extract_head(head, head_dim);

            // Compute gradients for queries, keys, and values
            let grad_query = grad_attention.dot(&model.output_attention_weights[head].transpose());
            let pred_head = predictions.extract_head(head, head_dim);
            let grad_key = grad_attention.transpose().dot(&pred_head);
            // Map grad_key back to embedding space
            let grad_key_reduced = pred_head.dot(&grad_key.transpose());

            // Compute value gradients
            let grad_value = grad_attention.dot(&model.output_attention_weights[head]);

            // Accumulate gradients for queries, keys, and values
            attention_gradients.add_head(&grad_query, head, head_dim);
            attention_gradients.add_head(&grad_key_reduced, head, head_dim);
            attention_gradients.add_head(&grad_value, head, head_dim);
        }

        attention_gradients
    }


    fn update_weights<T>(&self, model: &mut Model<T>, gradients: &Matrix, learning_rate: f64)
    where
        T: TaskTrait + TrainTrait,
    {
        // Update the final output weights for regression
        let output_gradients = gradients.dot(&model.final_output_weights.transpose());
        model.final_output_weights -= &output_gradients * learning_rate;

        // Update the feedforward network weights
        let ff_output_gradients = gradients.dot(&model.ff_output_weights.transpose());
        model.ff_output_weights -= &ff_output_gradients * learning_rate;

        let ff_hidden_gradients = gradients.dot(&model.ff_hidden_weights.transpose());
        model.ff_hidden_weights -= &ff_hidden_gradients * learning_rate;
    }

    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix 
    where 
        T: TaskTrait + TrainTrait
    {
        // Use the activation function stored in self.activation_fn
        let hidden = input.dot(&model.ff_hidden_weights)
            .apply(|x| model.apply_activation_fn(x)); // Apply the dynamic activation function here

        hidden.dot(&model.ff_output_weights)
    }


}


impl TrainTrait for ClassificationTaskImpl {

    fn compute_loss(&self, outputs: &Matrix, targets: &Matrix) -> f64 {

        let batch_size = outputs.rows as f64; // Number of samples
        outputs
            .rows_iter()
            .zip(targets.rows_iter())
            .map(|(predicted, target)| {
                -target.iter().zip(predicted.iter()).map(|(t, p)| t * p.ln()).sum::<f64>()
            })
            .sum::<f64>() / batch_size
    }

    fn compute_output_errors(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        // For classification, use softmax derivative
        let softmax_gradients = outputs.softmax_gradient(); // Derivative of softmax
        &softmax_gradients * (outputs - targets) // Element-wise multiplication
    }

    fn compute_accuracy(&self, outputs: &Matrix, labels: &[usize]) -> f64 {
        // Accuracy calculation for classification
        let correct_count: usize = outputs
            .rows_iter()
            .zip(labels.iter())
            .filter(|(predicted, &true_label)| Matrix::argmax_row(*predicted) == true_label)
            .count();

            if labels.is_empty() {
                return 0.0; // Prevent division by zero
            }

        (correct_count as f64 / labels.len() as f64) * 100.0 // Percentage accuracy
    }

    fn compute_final_output<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let logits = input.dot(&model.final_output_weights); // Corrected: Access `final_output_weights` from `model`
        logits.softmax() // Apply softmax for classification
    }

}

impl TrainTrait for RegressionTaskImpl {
    fn compute_loss(&self, outputs: &Matrix, targets: &Matrix) -> f64 {
        // Mean squared error for regression
        outputs
            .rows_iter()
            .zip(targets.rows_iter())
            .map(|(predicted, target)| {
                predicted.iter().zip(target.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>()
            })
            .sum::<f64>()
    }

    fn compute_output_errors(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        // For regression, output errors are the same as gradients
        outputs - targets
    }

    fn compute_accuracy(&self, _outputs: &Matrix, _labels: &[usize]) -> f64 {
        // Accuracy is not applicable for regression
        0.0
    }
    fn compute_final_output<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        input.dot(&model.final_output_weights) // (batch_size, 512) ⋅ (512, 1) → (batch_size, 1)
    }


}
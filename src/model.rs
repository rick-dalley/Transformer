// module Model
// Richard Dalley

use crate::activation_functions::{self};
use crate::matrix::{Matrix, Dot};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use serde_json::from_reader;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use std::time::Instant;
use std::io::{self, Write};

#[derive(Deserialize)]
pub struct Config {
    pub data_file: String,
    pub cap_data_rows: bool,
    pub max_data_rows: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub vocab_size: usize, // Size of the vocabulary for embedding
    pub shuffle_data: bool,
    pub validation_split: f64,
    pub sequence_length: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub model_dimensions: usize,
    pub hidden_dimensions: usize,
    pub columns: ColumnsConfig, // Add this field
    pub activation_fn_name: String,
    pub activation_alpha:f64,
    pub derivative_fn_name: String,
    pub derivative_alpha:f64,
    pub show_progress: bool,
}

#[derive(Deserialize)]
#[derive(Debug)] 
pub struct ColumnsConfig {
    pub features: Vec<String>,
    pub target: String,
    pub categorical_column: String,
}


// Model
pub struct Model {
    epochs: usize,
    show_progress: bool,
    cap_data_rows: bool,
    max_data_rows: usize,
    learning_rate: f64,
    shuffle_data: bool,
    validation_split: f64,
    sequence_length: usize,
    split_index: usize,
    batch_size: usize,
    data_location: String,
    num_layers: usize, 
    num_heads: usize, 
    embed_dim: usize, 
    vocab_size: usize, // Size of the vocabulary for embedding
    output_attention_weights: Vec<Matrix>,
    ff_hidden_weights: Matrix, // First linear layer weights
    ff_output_weights: Matrix, // Second linear layer weights
    final_output_weights: Matrix, 
    embedding_matrix: Matrix,
    columns: ColumnsConfig,
    data: Matrix,
    training_data:Matrix,
    validation_data: Matrix,
    labels: Vec<usize>,
    training_labels: Vec<usize>,
    validation_labels: Vec<usize>,
    activation_fn: Box<dyn activation_functions::ActivationTrait>,
    derivative_fn: Box<dyn activation_functions::ActivationTrait>,

}


impl Model {

    // from_json - build a model from json
    pub fn from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the JSON file
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Parse the JSON into a Config struct
        let config: Config = from_reader(reader)?;


        let activation_fn:Box<dyn activation_functions::ActivationTrait> = match config.activation_fn_name.to_lowercase().as_str() {
            "sigmoid" =>  Box::new(activation_functions::Sigmoid),
            "swish" =>  Box::new(activation_functions::Swish),
            "relu" =>  Box::new(activation_functions::ReLU),
            "leaky_relu" =>   Box::new(activation_functions::LeakyReLU { alpha: config.activation_alpha }),
            "elu" =>  Box::new(activation_functions::ELU { alpha: config.activation_alpha }),
            "tanh" =>  Box::new(activation_functions::TanH),
            _ => panic!("Unknown activation function: {}", config.activation_fn_name.to_lowercase().as_str()),
        };

        let derivative_fn:Box<dyn activation_functions::ActivationTrait> = match config.derivative_fn_name.to_lowercase().as_str() {
            "sigmoid_derivative" =>  Box::new(activation_functions::SigmoidDerivative),
            "relu_derivative" =>  Box::new(activation_functions::ReLUDerivative),
            "leaky_relu_derivative" =>  Box::new(activation_functions::LeakyReLUDerivative { alpha: config.derivative_alpha }),
            "elu_derivative" =>  Box::new(activation_functions::ELUDerivative { alpha: config.derivative_alpha }),
            _ => panic!("Unknown activation function: {}", config.derivative_fn_name.to_lowercase().as_str()),
        };

        // Initialize the Model struct
        let mut model = Self {
            epochs: config.epochs,
            cap_data_rows:config.cap_data_rows,
            max_data_rows: config.max_data_rows,
            learning_rate: config.learning_rate,
            shuffle_data: config.shuffle_data,
            vocab_size: config.vocab_size,
            validation_split: config.validation_split,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            batch_size: config.batch_size,
            embed_dim: config.model_dimensions,
            data_location: config.data_file.clone(),
            split_index: 0, // To be calculated during data processing
            sequence_length: config.sequence_length,
            columns: config.columns,
            output_attention_weights: (0..config.num_heads)
                .map(|_| Matrix::random(config.model_dimensions / config.num_heads, config.model_dimensions / config.num_heads))
                .collect(),            
            ff_hidden_weights: Matrix::random(config.model_dimensions, config.hidden_dimensions),
            ff_output_weights: Matrix::random(config.hidden_dimensions, config.model_dimensions),
            final_output_weights: Matrix::zeros(config.model_dimensions, config.model_dimensions),            // Initialize data-related fields
            data: Matrix::zeros(0, config.sequence_length), // Placeholder until loaded
            training_data: Matrix::zeros(0, config.sequence_length),
            validation_data: Matrix::zeros(0, config.sequence_length),
            labels: vec![],
            training_labels: vec![],
            validation_labels: vec![],
            embedding_matrix: Matrix::random(config.vocab_size, config.model_dimensions),
            show_progress: config.show_progress,
            activation_fn : activation_fn,
            derivative_fn : derivative_fn
        };

        // Assign data split index based on validation split
        model.split_index = ((1.0 - config.validation_split) * config.sequence_length as f64) as usize;
        // Initialize weights with suitable random distributions

        Ok(model)
    }

    // Use activation functions in methods
    pub fn apply_activation_fn(&self, x: f64) -> f64 {
        self.activation_fn.apply(x)
    }

    pub fn apply_derivative_fn(&self, x: f64) -> f64 {
        self.derivative_fn.apply(x)
    }



    pub fn load_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true) // Assume headers are present for column names
            .from_path(&self.data_location)?;

        // Extract the feature and target columns
        let feature_indices: Vec<usize> = reader
            .headers()?
            .iter()
            .enumerate()
            .filter(|(_, name)| self.columns.features.contains(&name.to_string()))
            .map(|(idx, _)| idx)
            .collect();

        let target_index = reader
            .headers()?
            .iter()
            .position(|name| name == &self.columns.target)
            .ok_or("Target column not found in the data file")?;

        let categorical_index = reader
            .headers()?
            .iter()
            .position(|name| name == &self.columns.categorical_column)
            .ok_or("Categorical column not found in the data file")?;

        // Initialize data storage
        let mut raw_data: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        let mut categorical_values: Vec<String> = Vec::new();

        let mut row_count = 0; // Track the number of rows processed

        // Read and process the data
        for record in reader.records() {
            if self.cap_data_rows && row_count >= self.max_data_rows {
                break; // Stop processing if cap_data_rows is enabled and max_data_rows is reached
            }

            let record = record?;

            // Extract features
            let features: Vec<f64> = feature_indices
                .iter()
                .map(|&idx| record[idx].parse::<f64>().unwrap_or(0.0))
                .collect();

            // Extract target
            let target: f64 = record[target_index].parse::<f64>()?;
            labels.push(target);

            // Extract categorical value
            categorical_values.push(record[categorical_index].to_string());

            raw_data.push(features);
            row_count += 1; // Increment the row count
        }
        // Normalize features dynamically (z-score normalization)
        let num_features = raw_data[0].len();
        let mut feature_means = vec![0.0; num_features];
        let mut feature_stds = vec![0.0; num_features];

        // Calculate mean and standard deviation for each feature
        for row in &raw_data {
            for (i, &value) in row.iter().enumerate() {
                feature_means[i] += value;
            }
        }
        feature_means.iter_mut().for_each(|mean| *mean /= raw_data.len() as f64);

        for row in &raw_data {
            for (i, &value) in row.iter().enumerate() {
                feature_stds[i] += (value - feature_means[i]).powi(2);
            }
        }
        feature_stds.iter_mut().for_each(|std| *std = (*std / raw_data.len() as f64).sqrt());

        // Apply z-score normalization
        for row in &mut raw_data {
            for (i, value) in row.iter_mut().enumerate() {
                *value = (*value - feature_means[i]) / feature_stds[i];
            }
        }

        // Handle categorical column (e.g., embedding or one-hot encoding)
        let categorical_map: std::collections::HashMap<String, usize> = categorical_values
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, value)| (value, idx))
            .collect();
        let categorical_indices: Vec<usize> = categorical_values
            .iter()
            .map(|value| *categorical_map.get(value).unwrap())
            .collect();

        // Create sequences
        let mut data: Vec<Vec<f64>> = Vec::new();
        let mut sequence_labels: Vec<f64> = Vec::new();

        for i in 0..(raw_data.len() - self.sequence_length) {
            let mut sequence: Vec<f64> = Vec::new();

            for j in 0..self.sequence_length {
                sequence.extend(&raw_data[i + j]); // Add features
                sequence.push(categorical_indices[i + j] as f64); // Add categorical index
            }

            data.push(sequence);
            sequence_labels.push(labels[i + self.sequence_length - 1]); // Use last value in sequence
        }

        // Convert data to Matrix format
        self.data = Matrix::new(data.len(), data[0].len(), data.into_iter().flatten().collect());
        self.labels = sequence_labels.iter().map(|&x| x as usize).collect();


        if self.validation_split > 0.0 {
            self.split_data();
        } else {
            // No split, assign all data to training
            self.split_index = self.data.rows;
            self.training_data = self.data.clone();
            self.training_labels = self.labels.clone();
        }

        Ok(())
    }

    pub fn shuffle_data(&mut self) {
        println!("Shuffling data...");

        // Generate shuffled indices
        let mut indices: Vec<usize> = (0..self.data.rows).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        // Create new shuffled matrices and labels
        let mut shuffled_data = Matrix::zeros(self.data.rows, self.data.cols);
        let mut shuffled_labels = Vec::with_capacity(self.labels.len());

        // Reorder data and labels based on shuffled indices
        for (new_idx, &original_idx) in indices.iter().enumerate() {
            // Copy row-by-row from the original data matrix
            for col in 0..self.data.cols {
                shuffled_data.data[new_idx * self.data.cols + col] =
                    self.data.data[original_idx * self.data.cols + col];
            }

            // Copy corresponding label
            shuffled_labels.push(self.labels[original_idx]);
        }

        // Update the data matrix and labels with shuffled versions
        self.data = shuffled_data;
        self.labels = shuffled_labels;

        println!("Data and labels shuffled.");
    }

    pub fn split_data(&mut self) {
        println!("Splitting data...");

        // Dynamically calculate split index based on validation_split
        self.split_index = ((1.0 - self.validation_split) * self.data.rows as f64) as usize;

        // Ensure split_index is valid
        assert!(
            self.split_index > 0 && self.split_index < self.data.rows,
            "Invalid split_index: {}. Ensure validation_split is correctly set.",
            self.split_index
        );

        // Split data into training and validation sets
        let data_cols = self.data.cols;

        // Extract rows for training data
        self.training_data = Matrix::new(
            self.split_index,
            data_cols,
            self.data.data[..(self.split_index * data_cols)].to_vec(),
        );

        // Extract rows for validation data
        self.validation_data = Matrix::new(
            self.data.rows - self.split_index,
            data_cols,
            self.data.data[(self.split_index * data_cols)..].to_vec(),
        );

        // Split labels into training and validation sets
        self.training_labels = self.labels[..self.split_index].to_vec();
        self.validation_labels = self.labels[self.split_index..].to_vec();

        println!(
            "Data split into training ({}) and validation ({}) sets.",
            self.training_data.rows, self.validation_data.rows
        );
    }

    //  computes query (Q), key (K), and value (V) matrices, and applies the attention formula:
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


    pub fn feedforward_network(&self, input: &Matrix) -> Matrix {
        // Use the activation function stored in self.activation_fn
        let hidden = input.dot(&self.ff_hidden_weights)
            .apply(|x| self.apply_activation_fn(x));  // Apply the dynamic activation function here
        
        hidden.dot(&self.ff_output_weights)
    }

    pub fn layer_norm(&self, input: &Matrix) -> Matrix {
        let epsilon = 1e-6;
        let mean = input.mean();
        let variance = input.variance();
        input.apply(|x| (x - mean) / (variance + epsilon).sqrt())
    }
        
    pub fn transformer_layer(&self, input: &Matrix) -> Matrix {
        // Multi-head attention with residual connection
        let attention_output = self.multi_head_attention(input, input, input, self.num_heads, self.embed_dim);

        // Create a temporary variable for the sum
        let residual_sum = input + &attention_output; // Add &attention_output to avoid moving it
        let attention_residual = self.layer_norm(&residual_sum);

        // Feedforward network with residual connection
        let ff_output = self.feedforward_network(&attention_residual);
        self.layer_norm(&(attention_residual + ff_output))
    }

    pub fn forward_transformer(&self, input: &Matrix) -> Matrix {
        // Extract the column with token indices (assume it's the first column for this example)
        let token_indices = input.column_to_indices(0); // Adjust the column index if needed

        // Apply embedding and positional encoding
        let mut x = self.embedding(&token_indices);
        let positional_enc = self.positional_encoding( self.embed_dim);
        x = x.add_broadcast(&positional_enc);

        // Pass through transformer layers
        for _ in 0..self.num_layers {
            x = self.transformer_layer(&x);
        }

        x
    }

    pub fn output_layer(&self, input: &Matrix) -> Matrix {
    let result = input.dot(&self.final_output_weights);  // Perform the dot product
    result.softmax()  // Apply softmax directly to the resulting matrix
}

    pub fn update_weights(&mut self, gradients: &Matrix, learning_rate: f64) {
        // Aggregate gradients across the batch to match weight dimensions
        let aggregated_gradients = gradients.mean_axis(0); // Collapse rows (result: 1 x 512)

        // Apply weight updates
        self.final_output_weights -= aggregated_gradients.broadcast(self.final_output_weights.rows) * learning_rate;
    }

    // backward_transformer
    pub fn backward_transformer(
        &self,
        predictions: &Matrix,
        output_errors: &Matrix,
    ) -> Matrix {
        // Initialize gradients
        let mut gradients = output_errors.clone(); // Start with output layer errors

        // Backpropagate through each transformer layer
        for _ in (0..self.num_layers).rev() {
            // Backpropagate through feedforward network
            gradients = self.backward_feedforward(&gradients);

            // Backpropagate through multi-head attention
            gradients = self.backward_multi_head_attention(&gradients, predictions);
        }

        gradients
    }

    fn backward_feedforward(&self, gradients: &Matrix) -> Matrix {
    // Derivatives through the second linear layer
    let grad_ff_output_weights = gradients.dot(&self.ff_hidden_weights);
    
    // Backpropagate activation function using the stored activation derivative function
    let grad_hidden = grad_ff_output_weights.apply(|x| self.apply_derivative_fn(x));

    grad_hidden.dot(&self.ff_hidden_weights.transpose())
}

    fn backward_multi_head_attention(
        &self,
        gradients: &Matrix,
        predictions: &Matrix,
    ) -> Matrix {
        let head_dim = self.embed_dim / self.num_heads;
        let mut attention_gradients = Matrix::zeros(gradients.rows, self.embed_dim);

        for head in 0..self.num_heads {
            // Extract gradients for this head
            let grad_attention = gradients.extract_head(head, head_dim);

            // Compute gradients for queries, keys, and values
            let grad_query = grad_attention.dot(&self.output_attention_weights[head].transpose());
            let pred_head = predictions.extract_head(head, head_dim);
            let grad_key = grad_attention.transpose().dot(&pred_head);
            // Map grad_key back to embedding space
            let grad_key_reduced = pred_head.dot(&grad_key.transpose());

            // Compute value gradients
    let grad_value = grad_attention.dot(&self.output_attention_weights[head]);

            // Accumulate gradients for queries, keys, and values
            attention_gradients.add_head(&grad_query, head, head_dim);
            attention_gradients.add_head(&grad_key_reduced, head, head_dim);
            attention_gradients.add_head(&grad_value, head, head_dim);
        }

        attention_gradients
    }

    pub fn train(&mut self) {
        let start_time = Instant::now(); // Track training time

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;

            // Shuffle training data at the start of each epoch
            self.shuffle_data();

            for i in (0..self.training_data.rows).step_by(self.batch_size) {
                // Prepare a batch of data
                let batch_data = self.training_data.slice(i, i + self.batch_size);
                let batch_labels = &self.training_labels[i..(i + self.batch_size).min(self.training_labels.len())];

                // Forward pass: transformer layers + output layer
                let predictions = self.forward_transformer(&batch_data);
                let outputs = self.output_layer(&predictions);

                // Compute loss (e.g., mean squared error)
                let mut batch_loss = 0.0;

                // Precompute target batch
                let target_batch = Matrix::from_labels(batch_labels, outputs.cols); 
                for (predicted_row, target_row) in outputs.rows_iter().zip(target_batch.rows_iter()) {
                    batch_loss += predicted_row
                        .iter()
                        .zip(target_row.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f64>() / outputs.cols as f64; // Mean Squared Error
                }

                total_loss += batch_loss;

                // Backward pass: calculate errors
                let output_errors = &outputs - &Matrix::from_labels(batch_labels, outputs.cols);
                let predictions_clone = predictions.clone();
                let attention_errors = self.backward_transformer(&predictions_clone, &output_errors);

                // === Resolve Immutable Borrow ===
                let correct_count: usize = outputs
                    .rows_iter()
                    .zip(batch_labels.iter()) // Resolve this borrow now
                    .filter(|(predicted, &true_label)| Matrix::argmax_row(*predicted) == true_label)
                    .count();
                
                correct_predictions += correct_count; // Done with `batch_labels` here

                // === Perform Mutable Borrow After Immutable Borrow Ends ===
                self.update_weights(&attention_errors, self.learning_rate); // No conflicts now

                // Show progress
                if self.show_progress && i % 1000 == 0 {
                    print!(".");
                    io::stdout().flush().unwrap();
                }
            }

            // Metrics for the epoch
            let accuracy = correct_predictions as f64 / self.training_data.rows as f64 * 100.0;
            println!(
                "Epoch {}/{} - Loss: {:.4}, Accuracy: {:.2}%",
                epoch + 1,
                self.epochs,
                total_loss / self.training_data.rows as f64,
                accuracy
            );
        }

        // Training completed
        let elapsed_time = start_time.elapsed();
        println!(
            "\nTraining completed in {:.2?} (hh:mm:ss.milliseconds)",
            elapsed_time
        );
    }

    pub fn print_config(&self) {
        println!("Model Configuration:");
        println!("  Epochs: {}", self.epochs);
        println!("  Learning Rate: {:.5}", self.learning_rate);
        println!("  Batch Size: {}", self.batch_size);
        println!("  Sequence Length: {}", self.sequence_length);
        println!("  Validation Split: {:.2}%", self.validation_split * 100.0);
        println!("  Split Index: {}", self.split_index);
        println!("  Shuffle Data: {}", self.shuffle_data);
        println!("  Number of Layers: {}", self.num_layers);
        println!("  Number of Heads: {}", self.num_heads);
        println!("  Embedding Dimension: {}", self.embed_dim);
        println!("  Vocabulary Size: {}", self.vocab_size);
        println!("  Data Location: {}", self.data_location);

        // Data dimensions
        println!("  Data Matrix: {} rows x {} cols", self.data.rows, self.data.cols);
        println!("  Training Data: {} rows x {} cols", self.training_data.rows, self.training_data.cols);
        if self.cap_data_rows {
            println!("  Capped for debugging for this run");
        }

        println!("  Validation Data: {} rows x {} cols", self.validation_data.rows, self.validation_data.cols);
        // Labels info
        println!("  Labels: {} total", self.labels.len());
        println!("  Training Labels: {} total", self.training_labels.len());
        println!("  Validation Labels: {} total", self.validation_labels.len());

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

}

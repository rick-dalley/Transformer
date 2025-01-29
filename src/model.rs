// module Model
// Richard Dalley

use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use crate::activation_functions::{self};
use crate::config;
use crate::matrix::{Matrix, Dot};
use crate::data_loader::DataLoader;


// Model
pub struct Model<'a> {
    pub data_loader: &'a mut DataLoader,
    epochs: usize,
    learning_rate: f64,
    classify: bool,
    batch_size: usize,
    num_layers: usize, 
    num_heads: usize, 
    embed_dim: usize, 
    output_attention_weights: Vec<Matrix>,
    ff_hidden_weights: Matrix, // First linear layer weights
    ff_output_weights: Matrix, // Second linear layer weights
    final_output_weights: Matrix, 
    embedding_matrix: Matrix,
    activation_fn: Box<dyn activation_functions::ActivationTrait>,
    derivative_fn: Box<dyn activation_functions::ActivationTrait>,
}


impl<'a> Model<'a> {
    

    // from_json - build a model from json
    pub fn from_json(config: &config::Config, data_loader:&'a mut DataLoader) -> Result<Self, Box<dyn std::error::Error>> {

        // Open the JSON file
        let config_clone = config.clone();
        let num_classes = 6;
        let epochs= config.epochs;
        let learning_rate =  config.learning_rate;
        let classify =  config.classify;
        let num_heads =  config.num_heads;
        let num_layers =  config.num_layers;
        let batch_size = config.batch_size;
        let embed_dim =  config.model_dimensions;
        let output_attention_weights =  (0..config.num_heads)
            .map(|_| Matrix::random(config.model_dimensions / config.num_heads, config.model_dimensions / config.num_heads))
            .collect();            
        let ff_hidden_weights =  Matrix::random(config.model_dimensions, config.hidden_dimensions);
        let ff_output_weights =  Matrix::random(config.hidden_dimensions, config.model_dimensions);
        let embedding_matrix =  Matrix::random(config.vocab_size, config.model_dimensions);

        let (activation_fn, derivative_fn) = activation_functions::get_activation_and_derivative(&config_clone);

        let final_output_weights = if config.classify {
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
            learning_rate,
            classify,
            num_heads,
            num_layers,
            batch_size,
            embed_dim,
            output_attention_weights,            
            ff_hidden_weights,
            ff_output_weights,
            final_output_weights,            // Initialize data-related fields
            embedding_matrix,
            activation_fn : activation_fn,
            derivative_fn : derivative_fn
        };


        Ok(model)
    }

    // Use activation functions in methods
    pub fn apply_activation_fn(&self, x: f64) -> f64 {
        self.activation_fn.apply(x)
    }

    pub fn apply_derivative_fn(&self, x: f64) -> f64 {
        self.derivative_fn.apply(x)
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
        // Aggregate gradients across the batch
        let aggregated_gradients = gradients.mean_axis(0); // (1, 512)

        if self.classify {
            // Classification: Final output weights are (512, num_classes)
            // Expand gradients to match (512, num_classes)
            let expanded_gradients = aggregated_gradients.broadcast(self.final_output_weights.cols); // (512, num_classes)
            self.final_output_weights -= expanded_gradients * learning_rate;
        } else {
            // Regression: Final output weights are (512, 1)
            // Transpose to match (512, 1)
            let transposed_gradients = aggregated_gradients.transpose(); // (512, 1)
            self.final_output_weights -= transposed_gradients * learning_rate;
        }
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

        if self.classify {
            self.train_classification();
        } else {
            self.train_regression();
        }
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

    pub fn train_classification(&mut self) {
        // Track training time
        let start_time = Instant::now();
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

            // Shuffle training data and labels
            DataLoader::shuffle_data(&mut self.data_loader.training_data, &mut self.data_loader.training_labels);

            for i in (0..self.data_loader.training_data.rows).step_by(self.batch_size) {
                // Prepare a batch of data
                let batch_data = self.data_loader.training_data.slice(i, i + self.batch_size);
                let batch_labels = &self.data_loader.training_labels[i..(i + self.batch_size).min(self.data_loader.training_labels.len())];

                // Forward pass
                let predictions = self.forward_transformer(&batch_data);
                let outputs = self.output_layer(&predictions);

                // Compute classification loss
                let target_batch = Matrix::from_labels(batch_labels, outputs.cols);
                let mut batch_loss = 0.0;
                for (predicted_row, target_row) in outputs.rows_iter().zip(target_batch.rows_iter()) {
                    batch_loss += predicted_row
                        .iter()
                        .zip(target_row.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f64>() / outputs.cols as f64;
                }
                total_loss += batch_loss;

                // Accuracy calculation
                let correct_count: usize = outputs
                    .rows_iter()
                    .zip(batch_labels.iter())
                    .filter(|(predicted, &true_label)| Matrix::argmax_row(*predicted) == true_label)
                    .count();
                correct_predictions += correct_count;

                // Backward pass
                let output_errors = &outputs - &target_batch;
                let predictions_clone = predictions.clone();
                let attention_errors = self.backward_transformer(&predictions_clone, &output_errors);
                self.update_weights(&attention_errors, self.learning_rate);

                // Update progress bar
                pb.inc(1);
            }

            let accuracy = correct_predictions as f64 / self.data_loader.training_data.rows as f64 * 100.0;
            println!(
                "Epoch {}/{} - Loss: {:.4}, Accuracy: {:.2}%",
                epoch + 1,
                self.epochs,
                total_loss / self.data_loader.training_data.rows as f64,
                accuracy
            );
        }

        let elapsed_time = start_time.elapsed();
        println!(
            "\nTraining completed in {:.2?} (hh:mm:ss.milliseconds)",
            elapsed_time
        );
    }

    pub fn train_regression(&mut self) {
        // Track training time
        let start_time = Instant::now();

        // Progress bar setup
        let iterations: u64 = (self.epochs * (self.data_loader.training_data.rows / self.batch_size)) as u64;
        let pb = ProgressBar::new(iterations);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            // Shuffle training data and labels
            DataLoader::shuffle_data(&mut self.data_loader.training_data, &mut self.data_loader.training_labels);

            for i in (0..self.data_loader.training_data.rows).step_by(self.batch_size) {
                // Prepare a batch of data
                let batch_data = self.data_loader.training_data.slice(i, i + self.batch_size);
                let batch_labels = &self.data_loader.training_labels[i..(i + self.batch_size).min(self.data_loader.training_labels.len())];

                // Forward pass
                let predictions = self.forward_transformer(&batch_data);
                let outputs = self.output_layer(&predictions);

                // Compute regression loss (MSE)
                let target_batch = Matrix::new(
                    batch_labels.len(),
                    1, // Single column for regression
                    batch_labels.iter().map(|&x| x as f64).collect(),
                );
                let mut batch_loss = 0.0;
                for (predicted_row, target_row) in outputs.rows_iter().zip(target_batch.rows_iter()) {
                    batch_loss += predicted_row
                        .iter()
                        .zip(target_row.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f64>() / outputs.cols as f64;
                }
                total_loss += batch_loss;

                // Backward pass
                let output_errors = &outputs - &target_batch;
                let expanded_output_errors = output_errors.repeat_columns(self.embed_dim);
                let predictions_clone = predictions.clone();
                let attention_errors = self.backward_transformer(&predictions_clone, &expanded_output_errors);
                self.update_weights(&attention_errors, self.learning_rate);

                // Update progress bar
                pb.inc(1);
            }

            println!(
                "Epoch {}/{} - Loss: {:.4}",
                epoch + 1,
                self.epochs,
                total_loss / self.data_loader.training_data.rows as f64
            );
        }

        let elapsed_time = start_time.elapsed();
        println!(
            "\nTraining completed in {:.2?} (hh:mm:ss.milliseconds)",
            elapsed_time
        );
    }

}

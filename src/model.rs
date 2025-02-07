// module Model
// Richard Dalley

// uses
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use crate::activation_functions::{self};
use crate::{config, data_loader};
use crate::matrix::{Matrix, Dot};
use crate::data_loader::DataLoader;
use crate::training_logs;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

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

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> (Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait;

    fn backward_transformer<T>(
        &self,
        model: &Model<T>,
        _outputs: &Matrix,
        predictions: &Matrix,
        output_errors: &Matrix,
        hidden_activations: &Matrix, // Add this argument
        input: &Matrix, // Add this argument
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait;

    fn backward_feedforward<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix, // Gradients from the next layer (dL/doutput)
        hidden_activations: &Matrix, // Hidden layer activations (before the second linear layer)
        input: &Matrix, // Input to the FFN
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
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

    
    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait;

    fn multi_head_attention<T>(
        &self, model: &Model<T>,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
        num_heads: usize,
        embed_dim: usize,
    ) -> Matrix
    where 
        T: TaskTrait + TrainTrait;

    fn scaled_dot_product_attention<T>(
        &self, model: &Model<T>, 
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
    ) -> Matrix 
    where 
        T: TaskTrait + TrainTrait;
     
}

pub trait TrainTrait {
    fn compute_loss(&self, outputs: &Matrix, targets: &Matrix) -> f64;
    fn compute_output_errors(&self, outputs: &Matrix, targets: &Matrix) -> Matrix;
    fn compute_accuracy(&self, outputs: &Matrix, labels: &data_loader::Labels) -> f64;
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
    logit_scaling_factor :f64,
    temperature_scaling: f64,
    batch_size: usize,
    num_layers: usize,
    num_heads: usize,
    embed_dim: usize,
    num_classes: usize,
    clipping_strategy: config::ClippingStrategy,
    clip_threshold: f64,
    output_attention_weights: Vec<Matrix>,
    query_weights: Matrix,
    key_weights: Matrix,
    value_weights: Matrix,
    query_bias: Matrix,
    key_bias: Matrix,
    value_bias: Matrix,
    ff_hidden_bias: Matrix,
    ff_output_bias: Matrix,
    final_output_bias: Matrix,
    ff_hidden_weights: Matrix,
    ff_output_weights: Matrix,
    final_output_weights: Matrix,
    embedding_matrix: Matrix,
    learning_task: config::LearningTask,
    activation_fn: Box<dyn activation_functions::ActivationTrait>,
    derivative_fn: Box<dyn activation_functions::ActivationTrait>,
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
        task: T, // Pass the task-specific implementation
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the JSON file
        let config_clone = config.clone();
        let num_classes = config.num_classes;
        let epochs = config.epochs;
        let checkpoint = config.checkpoint_interval;
        let learning_rate = config.learning_rate;
        let logit_scaling_factor = config.logit_scaling_factor;
        let temperature_scaling = config.temperature_scaling;
        let learning_task = config.learning_task;
        let clipping_strategy = config.clipping_strategy;
        let clip_threshold = config.clip_threshold;
        let num_heads = config.num_heads;
        let num_layers = config.num_layers;
        let batch_size = config.batch_size;
        let embed_dim = config.model_dimensions;
        let output_attention_weights = (0..config.num_heads)
            .map(|_| Matrix::random(config.model_dimensions / config.num_heads, config.model_dimensions / config.num_heads))
            .collect();

        let embedding_matrix = Matrix::random(config.vocab_size, config.model_dimensions);

        let (activation_fn, derivative_fn) = activation_functions::get_activation_and_derivative(&config_clone);

        let query_weights = Matrix::xavier(embed_dim, embed_dim);
        let key_weights = Matrix::xavier(embed_dim, embed_dim);
        let value_weights = Matrix::xavier(embed_dim, embed_dim);

        let ff_hidden_weights = Matrix::he(embed_dim, embed_dim); // Expand to 4 * embed_dim
        let ff_output_weights = Matrix::he(embed_dim, embed_dim); // Contract back to embed_dim

        let final_output_weights = if learning_task == config::LearningTask::Classification {
            Matrix::xavier(embed_dim, num_classes) // Classification
        } else {
            Matrix::xavier(embed_dim, 1) // Regression
        };

        let query_bias = Matrix::zeros(1, embed_dim);
        let key_bias = Matrix::zeros(1, embed_dim);
        let value_bias = Matrix::zeros(1, embed_dim);
        let ff_hidden_bias = Matrix::zeros(1, embed_dim); // Match hidden layer size
        let ff_output_bias = Matrix::zeros(1, embed_dim); // Match output size
        let final_output_bias = if learning_task == config::LearningTask::Classification {
            Matrix::zeros(1, num_classes) // Classification
        } else {
            Matrix::zeros(1, 1) // Regression
        };

        // Initialize the Model struct
        let model = Self {
            data_loader,
            epochs,
            checkpoint,
            learning_rate,
            logit_scaling_factor,
            temperature_scaling,
            learning_task,
            batch_size,
            num_heads,
            num_layers,
            embed_dim,
            num_classes,
            clipping_strategy,
            clip_threshold,
            output_attention_weights,
            query_weights,
            key_weights,
            value_weights,
            query_bias,
            key_bias,
            value_bias,
            ff_hidden_bias,
            ff_output_bias,
            final_output_bias,
            ff_hidden_weights,
            ff_output_weights,
            final_output_weights,
            embedding_matrix,
            activation_fn,
            derivative_fn,
            task, // Initialize task-specific behavior
        };

        Ok(model)
    }

    // Use activation functions in methods
    pub fn apply_activation_fn(&self, x: f64) -> f64 {
        self.activation_fn.apply(x)
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

    // positional_encoding
    pub fn positional_encoding(&self, seq_len: usize, embed_dim: usize) -> Matrix {
        let mut encoding = Matrix::zeros(seq_len, embed_dim); // Each row corresponds to a position

        for pos in 0..seq_len {
            for i in 0..embed_dim {
                let angle = pos as f64 / 10000f64.powf((i as f64) / (embed_dim as f64));
                if i % 2 == 0 {
                    encoding.data[pos * embed_dim + i] = angle.sin(); // Even indices: sine
                } else {
                    encoding.data[pos * embed_dim + i] = angle.cos(); // Odd indices: cosine
                }
            }
        }

        encoding
    }

    pub fn layer_norm(&self, input: &Matrix) -> Matrix {
        let epsilon = 1e-6;
        let mean = input.mean();
        let variance = input.variance();
        input.apply(|x| (x - mean) / (variance + epsilon).sqrt())
    }

    pub fn update_weights(&mut self, gradients: &Matrix) {
        
        let mut clip_threshold = self.clip_threshold;
        let aggregated_gradients =  match self.clipping_strategy {
            config::ClippingStrategy::None => {
                gradients.mean_axis(0)
            }
            config::ClippingStrategy::Static => {
                let clipped_gradients = gradients.apply(|g| g.max(-clip_threshold).min(clip_threshold));
                clipped_gradients.mean_axis(0)
            }
            config::ClippingStrategy::Dynamic => {
                let grad_norm = gradients.compute_norm();
                clip_threshold = grad_norm * 0.1;  // Clip at 10% of the norm
                let clipped_gradients = gradients.apply(|g| g.max(-clip_threshold).min(clip_threshold));
                clipped_gradients.mean_axis(0)

            }
        };

        // Aggregate gradients
        if self.learning_task == config::LearningTask::Classification {
            let expanded_gradients = aggregated_gradients.broadcast(self.final_output_weights.cols);

            if expanded_gradients.rows != self.final_output_weights.rows {
                self.final_output_weights -= expanded_gradients.transpose() * self.learning_rate;
            } else {
                self.final_output_weights -= expanded_gradients * self.learning_rate;
            }
        } else {
            let transposed_gradients = aggregated_gradients.transpose();
            self.final_output_weights -= transposed_gradients * self.learning_rate;
        }

    }

    pub fn train(&mut self) {
        // Track training time
        let start_time = Instant::now();
        let mut loss_history: Vec<f64> = Vec::new();
        let mut accuracy_history: Vec<f64> = Vec::new();

        // Progress bar setup
        let iterations: u64 = (self.epochs * (self.data_loader.training_data.rows / self.batch_size)) as u64;
        let pb = ProgressBar::new(iterations);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        // for the number of epochs asked for in config.json
        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            let mut total_samples = 0;
            // Shuffle training data and labels
            DataLoader::shuffle_data(&mut self.data_loader.training_data, &mut self.data_loader.training_labels);
            let rows = self.data_loader.training_data.rows;
            for i in (0..rows).step_by(self.batch_size) {
                // Prepare a batch of data
                let batch_data = self.data_loader.training_data.slice(i, i + self.batch_size);
                let batch_labels = self.data_loader.training_labels.slice(i, i + self.batch_size);

                // Forward pass
                let (predictions, hidden_activations) = self.task.forward_transformer(self, &batch_data);
    
                let outputs = self.task.compute_final_output(self, &predictions);

                let target_batch = match &batch_labels {
                    data_loader::Labels::Classification(vec) => Matrix::from_labels(vec, self.num_classes), // One-hot encode labels
                    data_loader::Labels::Regression(vec) => Matrix::new(vec.len(), 1, vec.clone()), // Use regression labels as is
                };


                let batch_loss = self.task.compute_loss(&outputs.clone(), &target_batch);
                total_loss += batch_loss;
                
                let accuracy = self.task.compute_accuracy(&outputs, &batch_labels);

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


                    let (
                        mut classification_errors,
                        grad_ff_hidden_weights,
                        grad_ff_hidden_bias,
                        grad_ff_output_weights,
                        grad_ff_output_bias
                    ) = self.task.backward_transformer(
                        self, &outputs, 
                        &predictions_clone, 
                        &expanded_output_errors, 
                        &hidden_activations, 
                        &batch_data.clone()
                    );             

                    self.ff_hidden_weights = grad_ff_hidden_weights * self.learning_rate;
                    self.ff_hidden_bias -= grad_ff_hidden_bias * self.learning_rate;
                    self.ff_output_weights -= grad_ff_output_weights * self.learning_rate;
                    self.ff_output_bias -=  grad_ff_output_bias * self.learning_rate;

                    classification_errors.clip_gradients_to(1.0);
                    classification_errors /= self.batch_size as f64;
                    self.update_weights(&classification_errors);  // Apply weight updates

                } else {
                    // Handle regression logic (unchanged)
                    let (
                        attention_errors, 
                        grad_hidden_weights,
                        grad_ff_hidden_bias,
                        grad_ff_output_weights,
                        grad_ff_output_bias
                    ) = self.task.backward_transformer(
                        self, 
                        &outputs, 
                        &predictions_clone, 
                        &expanded_output_errors, 
                        &hidden_activations, 
                        &batch_data.clone()
                    );
                    self.ff_hidden_bias -= grad_ff_hidden_bias * self.learning_rate;
                    self.ff_output_weights -= grad_ff_output_weights * self.learning_rate;
                    self.ff_output_bias -=  grad_ff_output_bias * self.learning_rate;
                    self.update_weights( &attention_errors);
                }

                // Update progress bar
                pb.inc(1);
            } // batch


            // Record the results of the run
            training_logs::log_epoch_results(
                "log_files/training_log.csv",
                epoch,
                total_loss,
                correct_predictions,
                total_samples,
                self.final_output_weights.clone(),
                &mut loss_history,
                &mut accuracy_history,
                rows,
            );

           println!("Epoch {}/{} - Loss: {:.4}", epoch + 1, self.epochs, total_loss / rows as f64);

            // Perform periodic checkpointing
            if epoch % self.checkpoint == 0 {
                println!("Saving model checkpoint...");
                if let Err(e) = self.save_checkpoint("log_files/model_checkpoint.json") {
                    eprintln!("Error saving model checkpoint: {}", e);
                }
            }

        } //epoch

        let elapsed_time = start_time.elapsed();
        println!(
            "\nTraining completed in {:.2?} (hh:mm:ss.milliseconds)",
            elapsed_time
        );
    }
    
    // save_checkpoint
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

        // Data Info
        println!("  Data Location: {}", self.data_loader.data_location);
        println!("  Training Data: {} rows x {} cols", self.data_loader.training_data.rows, self.data_loader.training_data.cols);
        if self.data_loader.cap_data_rows {
            println!("  Capped for debugging for this run");
        }
        println!("  Training Labels: {} total", self.data_loader.training_labels.len());
        println!("  Sequence Length: {}", self.data_loader.sequence_length);

        println!("  Split Index: {}", self.data_loader.split_index);

        println!("  Validation Data: {} rows x {} cols", self.data_loader.validation_data.rows, self.data_loader.validation_data.cols);
        println!("  Validation Split: {:.2}%", self.data_loader.validation_split * 100.0);
        println!("  Validation Labels: {} total", self.data_loader.validation_labels.len());

        // Labels info
        println!("  Epochs: {}", self.epochs);
        println!("  Learning Rate: {:.5}", self.learning_rate);
        println!("  Batch Size: {}", self.batch_size);
        println!("  Number of Layers: {}", self.num_layers);
        println!("  Number of Heads: {}", self.num_heads);
        println!("  Embedding Dimension: {}", self.embed_dim);

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

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> (Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.forward_transformer(model, input),
            TaskEnum::Regression(task) => task.forward_transformer(model, input),
        }
    }

    fn backward_transformer<T>(
        &self,
        model: &Model<T>,
        _outputs: &Matrix,
        predictions: &Matrix,
        output_errors: &Matrix,
        hidden_activations: &Matrix, // Add this argument
        input: &Matrix, // Add this argument
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait
    {
        match self {
            TaskEnum::Classification(task) => task.backward_transformer(model, _outputs, predictions, output_errors, hidden_activations, input),
            TaskEnum::Regression(task) => task.backward_transformer(model, _outputs, predictions, output_errors, hidden_activations, input),
        }
    }

    fn backward_feedforward<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix, // Gradients from the next layer (dL/doutput)
        hidden_activations: &Matrix, // Hidden layer activations (before the second linear layer)
        input: &Matrix, // Input to the FFN
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait
    {
        match self {
            TaskEnum::Classification(task) => task.backward_feedforward(model, gradients, hidden_activations, input),
            TaskEnum::Regression(task) => task.backward_feedforward(model, gradients, hidden_activations, input),
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

    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.feedforward_network(model, input),
            TaskEnum::Regression(task) => task.feedforward_network(model, input),
        }
    }

 
    fn multi_head_attention<T>(
        &self, model: &Model<T>,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
        num_heads: usize,
        embed_dim: usize,
    ) -> Matrix
    where 
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.multi_head_attention(model, query, key, value, num_heads, embed_dim),
            TaskEnum::Regression(task) => task.multi_head_attention(model, query, key, value, num_heads, embed_dim),
        }
    }

    fn scaled_dot_product_attention<T>(
        &self, model: &Model<T>, 
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
    ) -> Matrix 
    where 
        T: TaskTrait + TrainTrait,
    {
        match self {
            TaskEnum::Classification(task) => task.scaled_dot_product_attention(model,query, key, value),
            TaskEnum::Regression(task) => task.scaled_dot_product_attention(model, query, key, value),
        }
        
    }

}


// Implement TaskTrait for ClassificationTask and RegressionTask
pub struct ClassificationTaskImpl;

impl TaskTrait for ClassificationTaskImpl {

    fn scaled_dot_product_attention<T>(
        &self, _model: &Model<T>,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
    ) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let d_k = query.cols as f64;

        let scores = query.dot(&key.transpose()) / (d_k.sqrt() + 1e-8);

        let attention_weights = scores.softmax();

        attention_weights.dot(value)
    }

   fn multi_head_attention<T>(
        &self, model: &Model<T>,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
        num_heads: usize,
        embed_dim: usize,
    ) -> Matrix 
     where
        T: TaskTrait + TrainTrait,
    {
        let head_dim = embed_dim / num_heads;
        let mut attention_heads = Vec::new();

        for head in 0..num_heads {
            let q = query.extract_head(head, head_dim);
            let k = key.extract_head(head, head_dim);
            let v = value.extract_head(head, head_dim);
            let scaled_attention = model.task.scaled_dot_product_attention(model, &q, &k, &v);

            // Classification applies learned attention weights
            let transformed = scaled_attention.dot(&model.output_attention_weights[head]);

            attention_heads.push(transformed);
        }

        Matrix::concat_heads(&attention_heads)
    }

    fn transformer_layer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait, 
    {
        let attention_output = model.task.multi_head_attention(model,input, input, input, model.num_heads, model.embed_dim);
        let residual_sum = input + &attention_output; 
        let attention_residual = model.layer_norm(&residual_sum); 
        
        let ff_output = model.task.feedforward_network(model, &attention_residual);
        let ff_residual = attention_residual + ff_output; // Ensure residual connection
        
        model.layer_norm(&ff_residual) 
    }

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> (Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {
        let token_indices = input.column_to_indices(0);
        let mut x = model.embedding(&token_indices);

        let positional_enc = model.positional_encoding(input.rows, model.embed_dim);
        let hidden_activations = x.clone();
        x = x.add_broadcast(&positional_enc);

        for _ in 0..model.num_layers {
            x = self.transformer_layer(model, &x);
        }
        (x, hidden_activations)
    }

    fn backward_transformer<T>
    (
        &self,
        model: &Model<T>,
        outputs: &Matrix,
        predictions: &Matrix,
        output_errors: &Matrix,
        hidden_activations: &Matrix, // Add this argument
        input: &Matrix, // Add this argument
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {
    
        // Compute the gradient of the softmax output
        let softmax_gradients = outputs.softmax_gradient(output_errors); // Derivative of softmax

        // Multiply by the output errors to get the gradients
        let gradients = &softmax_gradients * output_errors;

        // Backpropagate through the feedforward networka
        let (
            ff_gradients, 
            grad_ff_hidden_bias, 
            grad_ff_output_weights, 
            grad_ff_output_bias, 
            grad_input
        ) = self.backward_feedforward(model, &gradients, hidden_activations, input);

        // Backpropagate through the multi-head attention
        let attention_gradients = self.backward_multi_head_attention(model, &ff_gradients, predictions);

        // Return the final gradients
        (attention_gradients, ff_gradients, grad_ff_hidden_bias, grad_ff_output_bias, grad_ff_output_weights)
    }


 
    fn backward_feedforward<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix, // Gradients from the next layer (dL/doutput)
        hidden_activations: &Matrix, // Hidden layer activations (before the second linear layer)
        input: &Matrix, // Input to the FFN
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {
        // Gradients for the second linear layer (dL/dW2)
        let grad_ff_output_weights = hidden_activations.transpose().dot(&gradients);

        // Gradients for the output biases (dL/db2)
        let grad_ff_output_bias = gradients.sum_rows(); // Sum over the batch dimension

        // Gradients through the second linear layer (dL/dhidden)
        let grad_hidden = gradients.dot(&model.ff_output_weights.transpose());

        // Gradients through the activation function (dL/dpre_activation)
        let grad_pre_activation = grad_hidden.apply(|x| model.derivative_fn.apply(x));

        // Gradients for the first linear layer (dL/dW1)
        let grad_ff_hidden_weights = input.transpose().dot(&grad_pre_activation);

        // Gradients for the hidden biases (dL/db1)
        let grad_ff_hidden_bias = grad_pre_activation.sum_rows(); // Sum over the batch dimension

        // Gradients propagated to the previous layer (dL/dinput)
        let grad_input = grad_pre_activation.dot(&model.ff_hidden_weights.transpose());

        // Return all computed gradients
        (
            grad_ff_hidden_weights, // dL/dW1
            grad_ff_hidden_bias,    // dL/db1
            grad_ff_output_weights, // dL/dW2
            grad_ff_output_bias,    // dL/db2
            grad_input
        )
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


    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let hidden = input.dot(&model.ff_hidden_weights) 
            .apply(|x| model.apply_activation_fn(x));
        
        hidden.dot(&model.ff_hidden_weights.transpose()) 
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

    fn compute_accuracy(&self, outputs: &Matrix,  labels: &data_loader::Labels) -> f64 {
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

    fn scaled_dot_product_attention<T>(
        &self, _model: &Model<T>,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
    ) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        let d_k = query.cols as f64;

        let scores = query.dot(&key.transpose()) / (d_k.sqrt() + 1e-8);

        let attention_weights = scores.softmax_rows();

        let scaled_attention = attention_weights.dot(value);

        scaled_attention
    }

   fn multi_head_attention<T>(
        &self, model: &Model<T>,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
        num_heads: usize,
        embed_dim: usize,
    ) -> Matrix 
     where
        T: TaskTrait + TrainTrait,
    {
        let head_dim = embed_dim / num_heads;

        let mut attention_heads = Vec::new();

        for head in 0..num_heads {
            // extract q, k, v
            let q = query.extract_head(head, head_dim);
            let k = key.extract_head(head, head_dim);
            let v = value.extract_head(head, head_dim);

            let scaled_attention = model.task.scaled_dot_product_attention(model, &q, &k, &v);


            // Regression does NOT apply learned weights—raw attention scores are used
            let transformed = scaled_attention.clone();

            attention_heads.push(transformed);
        }

        Matrix::concat_heads(&attention_heads)
    }


    ////////////////
    /// Transformer
    fn transformer_layer<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {
        // Compute query, key, and value
        let query = input.dot(&model.query_weights) + model.query_bias.broadcast(input.rows);
        let key = input.dot(&model.key_weights) + model.key_bias.broadcast(input.rows);
        let value = input.dot(&model.value_weights) + model.value_bias.broadcast(input.rows);

        // Multi-head attention
        let attention_output = model.task.multi_head_attention(
            model, &query, &key, &value, model.num_heads, model.embed_dim,
        );

        // Residual connection and layer normalization
        let residual_sum = input + &attention_output;
        let attention_residual = model.layer_norm(&residual_sum);

        // Feedforward network
        let ff_output = model.task.feedforward_network(model, &attention_residual);

         // Residual connection + LayerNorm (Post-FFN)
        let output = &attention_residual + &ff_output;
        let output = model.layer_norm(&output);

        output
    }

    fn forward_transformer<T>(&self, model: &Model<T>, input: &Matrix) -> (Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {

        let token_indices = input.column_to_indices(0);

        let mut x = model.embedding(&token_indices);

        let positional_enc = model.positional_encoding(input.rows, model.embed_dim);

        x += positional_enc;
    
        let hidden_activations = x.clone(); // Store activations before 2nd linear layer

        for _ in 0..model.num_layers {
            x = self.transformer_layer(model, &x);
        }

        (x, hidden_activations)
    }

    fn backward_transformer<T>(
        &self,
        model: &Model<T>,
        _outputs: &Matrix,
        predictions: &Matrix,
        output_errors: &Matrix,
        hidden_activations: &Matrix, // Add this argument
        input: &Matrix, // Add this argument
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {
        // For regression, the output errors are the same as the gradients
        let gradients = output_errors.clone();

        // Backpropagate through the feedforward network
        let (
            grad_ff_hidden_weights, 
            grad_ff_hidden_bias, 
            grad_ff_output_weights, 
            grad_ff_output_bias,
            grad_input
        ) = self.backward_feedforward(model, &gradients, hidden_activations, input);

        // Backpropagate through the multi-head attention
        let attention_gradients = self.backward_multi_head_attention(model, &grad_input, predictions);

        // Return the final gradients
        (attention_gradients, grad_ff_hidden_weights, grad_ff_hidden_bias, grad_ff_output_weights, grad_ff_output_bias)
    }

    fn backward_feedforward<T>(
        &self,
        model: &Model<T>,
        gradients: &Matrix, // Gradients from the next layer (dL/doutput)
        hidden_activations: &Matrix, // Hidden layer activations (before the second linear layer)
        input: &Matrix, // Input to the FFN
    ) -> (Matrix, Matrix, Matrix, Matrix, Matrix)
    where
        T: TaskTrait + TrainTrait,
    {
        // Gradients for the second linear layer (dL/dW2)
        let grad_ff_output_weights = hidden_activations.transpose().dot(&gradients);

        // Gradients for the output biases (dL/db2)
        let grad_ff_output_bias = gradients.sum_rows(); // Sum over the batch dimension

        // Gradients through the second linear layer (dL/dhidden)
        let grad_hidden = gradients.dot(&model.ff_output_weights.transpose());

        // Gradients through the activation function (dL/dpre_activation)
        let grad_pre_activation = grad_hidden.apply(|x| model.derivative_fn.apply(x));

        // Gradients for the first linear layer (dL/dW1)
        let grad_ff_hidden_weights = input.transpose().dot(&grad_pre_activation);

        // Gradients for the hidden biases (dL/db1)
        let grad_ff_hidden_bias = grad_pre_activation.sum_rows(); // Sum over the batch dimension

        // Gradients propagated to the previous layer (dL/dinput)
        let grad_input = grad_pre_activation.dot(&model.ff_hidden_weights.transpose());

        // Return all computed gradients
        (
            grad_ff_hidden_weights, // dL/dW1
            grad_ff_hidden_bias,    // dL/db1
            grad_ff_output_weights, // dL/dW2
            grad_ff_output_bias,    // dL/db2
            grad_input
        )
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

    // feedforward_network
    fn feedforward_network<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix 
    where 
        T: TaskTrait + TrainTrait
    {
        // First linear transformation: input (batch_size, model_dim) -> hidden (batch_size, hidden_dim)
        let hidden = input.dot(&model.ff_hidden_weights) + model.ff_hidden_bias.broadcast(input.rows);
        
        // Activation function
        let hidden = hidden.map(|x| x.max(0.0)); // ReLU

        // Second linear transformation: hidden (batch_size, hidden_dim) -> output (batch_size, model_dim)
        let output = hidden.dot(&model.ff_output_weights) + model.ff_output_bias.broadcast(input.rows);

        output // (32 × 512), ensuring residual connection compatibility
    }
    
}


impl TrainTrait for ClassificationTaskImpl {

    fn compute_loss(&self, logits: &Matrix, targets: &Matrix) -> f64 {
        let log_probs = logits.log_softmax();  // Convert logits to log probabilities
        let batch_size = log_probs.rows as f64; // Number of samples
        log_probs
            .rows_iter()
            .zip(targets.rows_iter())
            .map(|(log_prob, target)| {
                -target.iter().zip(log_prob.iter()).map(|(t, lp)| t * lp).sum::<f64>()
            })
            .sum::<f64>() / batch_size
    }

    fn compute_output_errors(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        
        outputs - targets // Element-wise multiplication
    }

    fn compute_accuracy(&self, outputs: &Matrix,  labels: &data_loader::Labels) -> f64 {
        // Accuracy calculation for classification
        let classification_labels = match labels {
            data_loader::Labels::Classification(vec) => vec,
            _ => panic!("compute_accuracy() called with non-classification labels"),
        };

        let correct_count: usize = outputs
            .rows_iter()
            .zip(classification_labels.iter()) // Use extracted `Vec<usize>`
            .filter(|(predicted, &true_label)| Matrix::argmax_row(*predicted) == true_label)
            .count();

        if classification_labels.is_empty() {
            return 0.0; // Prevent division by zero
        }

        (correct_count as f64 / classification_labels.len() as f64) * 100.0 
    }

    fn compute_final_output<T>(&self, model: &Model<T>, input: &Matrix) -> Matrix
    where
        T: TaskTrait + TrainTrait,
    {

            input.dot(&model.final_output_weights)
            .apply(|x| x * model.logit_scaling_factor * model.temperature_scaling) 
            
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

    fn compute_accuracy(&self, _outputs: &Matrix,  _labels: &data_loader::Labels) -> f64 {
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
use std::io::Write;
use std::fs::{File, OpenOptions};
use std::sync::Once;

use crate::matrix::Matrix;

pub fn log_epoch_results(
    log_location: &str,
    epoch: usize,
    total_loss: f64,
    correct_predictions: usize,
    total_samples: usize,
    weights: Matrix,
    loss_history: &mut Vec<f64>,
    accuracy_history: &mut Vec<f64>,
    training_data_rows: usize,
) {
    // Compute average loss and accuracy
    let avg_loss = total_loss / training_data_rows as f64;
    let accuracy = (correct_predictions as f64 / total_samples as f64) * 100.0;

    // Store values for analysis
    loss_history.push(avg_loss);
    accuracy_history.push(accuracy);

    // Compute mean and std dev of final_output_weights directly inside this function
    let weights_data = &weights.data;
    let weight_mean = weights_data.iter().sum::<f64>() / weights_data.len() as f64;
    let weight_std = (weights_data.iter().map(|w| (w - weight_mean).powi(2)).sum::<f64>() / weights_data.len() as f64).sqrt();

    // Log everything
    log_training_metrics(epoch, avg_loss, accuracy, weight_mean, weight_std, log_location);
}

// log training metrics
pub fn log_training_metrics(
    epoch: usize, 
    loss: f64, 
    accuracy: f64, 
    weight_mean: f64, 
    weight_std: f64, 
    log_location: &str
) {
    static HEADER_PRINTED: Once = Once::new();
    
    
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    HEADER_PRINTED.call_once(|| {
        writeln!(file, "epoch, avg_loss, accuracy, weight_mean, weight_std")
            .expect("Failed to write header.");
    });
    
    writeln!(file, "{},{},{},{},{}", epoch, loss, accuracy, weight_mean, weight_std)
        .expect("Failed to write log.");
}

pub fn log_weights(epoch: usize, iteration:usize, tag:&str, weights: Matrix, log_location: &str){
    
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");
    let mean = weights.mean();
    let std = weights.std_dev();
    let min = weights.min();
    let max = weights.max();
    let sum = weights.sum();
    
   // Handle potential errors from `writeln!`
    if let Err(e) = writeln!(file, "{},{},{},{},{},{},{}, {}", epoch, iteration, tag, mean, std, min, max, sum) {
        eprintln!("Failed to write to log file: {}", e);
    }
}


pub fn log_softmax_gradient(epoch: usize, iteration:usize, gradients: Matrix, output_errors: Matrix, log_location: &str){
    
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    let softmax_gradients = gradients.softmax_gradient(&output_errors);
    let norm =  softmax_gradients.compute_norm();
   // Handle potential errors from `writeln!`
    if let Err(e) = writeln!(file, "{},{},{}", epoch, iteration, norm) {
        eprintln!("Failed to write to log file: {}", e);
    }
}


pub fn log_gradient_norms(grad_norms: &Vec<f64>, filename: &str) {
    let mut file = File::create(filename).expect("Could not create file");

    for (step, norm) in grad_norms.iter().enumerate() {
        writeln!(file, "{},{}", step, norm).expect("Could not write to file");
    }
}

pub fn log_matrix_norms(epoch:usize, iteration:usize, matrix: Matrix, log_location:&str){
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");


        writeln!(file, "{}, {},{}", epoch, iteration, matrix.compute_norm()).expect("Could not write to file");

}

pub fn log_weights_update(
    grad_norm: f64, 
    clip_threshold: f64, 
    learning_rate: f64, 
    aggregated_gradients: Matrix, 
    final_output_weights: Matrix, 
    expanded_gradients: Matrix, 
    final_output_weights_after: Matrix) {
       let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log_files/log_weights_update.csv")
        .expect("Failed to open log file");
 
    writeln!(file, "{}, {},{:?}, {}, {:?}, {:?}, {:?}", grad_norm, clip_threshold, aggregated_gradients.data.get(0..5),  learning_rate, final_output_weights.data.get(0..5), expanded_gradients.data.get(0..5), final_output_weights_after.data.get(0..5)).expect("Could not write to file");
}

pub fn log_update_weights_min_max_means(grad_norm:f64, gradients:Matrix, clipped_gradients:Matrix, aggregated_gradients: Matrix){
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log_files/log_min_max_mean.csv")
        .expect("Failed to open log file");
    static HEADER_PRINTED: Once = Once::new();

    HEADER_PRINTED.call_once(|| {
        writeln!(file, "grad_norm, gradients:min, max, mean, clipped_gradients:min, max, mean, aggregated_gradients:min, max, mean")
            .expect("Failed to write header.");
    });
    writeln!(file, "{}, {},{}, {}, {}, {}, {}, {}, {}, {}",  grad_norm, gradients.min(), gradients.max(), gradients.mean(), clipped_gradients.min(), clipped_gradients.max(), clipped_gradients.mean(), aggregated_gradients.min(), aggregated_gradients.max(), aggregated_gradients.mean()).expect("Could not write to file");
}

pub fn log_weight_changes(gradients:Matrix, before: Matrix, after: Matrix){
        let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log_files/weight_change.csv")
        .expect("Failed to open log file");

    let delta: f64 = after.data.iter().zip(before.data.iter()).map(|(new, old)| (new - old).abs()) 
        .sum();

    let grad_mean = gradients.mean();
    let grad_std = gradients.std_dev();
    let grad_min = gradients.min();
    let grad_max = gradients.max();
    static HEADER_PRINTED: Once = Once::new();

    HEADER_PRINTED.call_once(|| {
        writeln!(file, "delta, gradients:mean, std, min, max")
            .expect("Failed to write header.");
    });

    writeln!(file, "{}, {}, {}, {}, {}",  delta, grad_mean, grad_std, grad_min, grad_max).expect("Could not write to file");
}

pub fn log_activations(hidden:Matrix) {
        // **Log Activations**
    let mean_activation = hidden.mean();
    let std_activation = hidden.std_dev();
    let min_activation = hidden.min();
    let max_activation = hidden.max();

    // Write to a log file
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log_files/activations.csv")
        .expect("Failed to open activations log file");

    static HEADER_PRINTED: Once = Once::new();
    HEADER_PRINTED.call_once(|| {
        writeln!(file, "mean, std, min, max").expect("Failed to write header.");
    });

    writeln!(file, "{}, {}, {}, {}", mean_activation, std_activation, min_activation, max_activation)
        .expect("Could not write activation stats");
}
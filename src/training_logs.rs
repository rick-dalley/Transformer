use std::io::Write;
use std::fs::OpenOptions;
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

pub fn log_matrix_norms(epoch:usize, iteration:usize, matrix: Matrix, log_location:&str){
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");


        writeln!(file, "{}, {},{}", epoch, iteration, matrix.compute_norm()).expect("Could not write to file");

}


pub fn log_matrix_stats(epoch:usize, iteration:usize, matrix:Matrix, log_location: &str, name:&str, squelch:bool) {
    if squelch {
        return;
    }
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");
        


    static HEADER_PRINTED: Once = Once::new();
    HEADER_PRINTED.call_once(|| {
        writeln!(file, "name, epoch, iteration, norm, mean, std, min, max").expect("Failed to write header.");
    });

    writeln!(file, "{}, {}, {}, {}, {}, {}, {}, {}", name, epoch, iteration, matrix.compute_norm(), matrix.mean(), matrix.std_dev(), matrix.min(), matrix.max()).expect("Could not write to file");

}

pub fn log_n_elements(name:&str, slice:&Vec<f64>, n_elements:usize, log_location: &str){

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    writeln!(file, "{} first {}, {:?}", name,n_elements, &slice[..n_elements.min(slice.len())] ).expect("Could not write to file");

}

pub fn log_sample(name: &str, rows:usize, n_elements:usize, matrix:Matrix, log_location:&str, squelch:bool){
    if squelch {
        return;
    }
    for i in 0..rows.min(matrix.rows) {
        let row_slice = matrix.sample(i, n_elements); 
        log_n_elements(name, &row_slice, rows, log_location);
    }
}
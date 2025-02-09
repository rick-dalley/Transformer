use std::io::Write;
use std::fs::OpenOptions;
use std::sync::Once;
use crate::matrix::Matrix;

// log training metrics
pub fn log_training_metrics(
    epoch: usize, 
    loss: f64, 
    accuracy: f64, 
    std: f64,
    mean: f64,
    log_location: &str,
    squelch: bool
) {

    if squelch {
        return;
    }
    static HEADER_PRINTED: Once = Once::new();
    
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    HEADER_PRINTED.call_once(|| {
        writeln!(file, "epoch, avg_loss, accuracy, final_output_weight_std, final_output_weight_mean, ff_output_weights_std, ff_output_weights_mean")
            .expect("Failed to write header.");
    });

    writeln!(file, "{}, {},{}, {}, {}", epoch, loss, accuracy, std, mean)
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
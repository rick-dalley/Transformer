
use serde::Deserialize;

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
    pub activation_lambda:f64,
    pub show_progress: bool,
}

#[derive(Deserialize)]
#[derive(Debug)] 
pub struct ColumnsConfig {
    pub features: Vec<String>,
    pub target: String,
    pub categorical_column: String,
}


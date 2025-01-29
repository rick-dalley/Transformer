use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Error as IoError};
use serde_json::from_reader;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub data_source:String,
    pub connection_string:String,
    pub location: String,
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
    pub classify: bool,
    pub model_dimensions: usize,
    pub hidden_dimensions: usize,
    pub columns: ColumnsConfig, // Add this field
    pub activation_fn_name: String,
    pub activation_alpha:f64,
    pub activation_lambda:f64,
}

#[derive(Deserialize,Debug, Clone)] 
pub struct ColumnsConfig {
    pub features: Vec<String>,
    pub target: String,
    pub categorical_column: String,
}

impl Config {
    pub fn from_json(path: &str) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = from_reader(reader)?;
        Ok(config)
    }
}
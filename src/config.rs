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
    pub check_points: usize,
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
    pub activation_fn_name: String,
    pub activation_alpha:f64,
    pub activation_lambda:f64,

    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize, // Set dynamically

    //optional if there is no column header then this is not needed
    pub columns: Option<ColumnsConfig>, 
}

#[derive(Deserialize,Debug, Clone)] 
pub struct ColumnsConfig {
    pub features: Vec<String>,
    pub target: String,
    pub categorical_column: String,
}

// Default function for Serde (to ensure it's always present)
fn default_checkpoint_interval() -> usize {
    1 // Default to 1 if missing
}

impl Config {
    // get config from the json file
    pub fn from_json(path: &str) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut config: Config = from_reader(reader)?;
        config.checkpoint_interval = Config::checkpoint_interval(config.epochs, config.check_points);

        if config.columns.is_none() {
            println!("Warning, no columns were defined.  Assuming fixed feature set.")
        }
        Ok(config)
    }
    //checkpoint calculation
    pub fn checkpoint_interval(epochs: usize, check_points:usize) -> usize{
        let mut checkpoints = check_points;
        if checkpoints < 1 || checkpoints > epochs{
            checkpoints = if epochs > 1 {epochs / 2} else {1};
        }
        epochs / checkpoints
    }
}
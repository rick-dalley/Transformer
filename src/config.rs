use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Error as IoError};
use serde_json::from_reader;

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum LearningTask {
    Classification,
    Regression,
    Unsupervised,
}

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
    #[serde(default = "default_logit_scaling")]
    pub logit_scaling_factor :f64,

    #[serde(default = "default_clip_threshold")]
    pub clip_threshold: f64,

    #[serde(default = "default_temperature_scaling")]
    pub temperature_scaling: f64,

    pub vocab_size: usize, // Size of the vocabulary for embedding
    pub shuffle_data: bool,
    pub validation_split: f64,

    #[serde(default = "default_learning_task")]
    pub learning_task: LearningTask,
    pub learning: String,

    #[serde(default = "default_sequence_data")]
    pub sequence_data: bool,
    pub sequence_length: usize,
    pub batch_size: usize,

    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    pub num_heads: usize,
    pub num_layers: usize,

    #[serde(default = "default_label_index")]
    pub label_column_index: usize,

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

fn default_num_classes() -> usize {
    1
}

// Default function for Serde (to ensure it's always present)
fn default_sequence_data() -> bool {
    false // Default to 1 if missing
}

// Default function for Serde (to ensure it's always present)
fn default_label_index() -> usize {
    usize::MAX // Default to 1 if missing
}

// Default function for Serde (to ensure it's always present)
fn default_learning_task() -> LearningTask {
    LearningTask::Regression
}

fn default_clip_threshold() -> f64 {
    1.0
}

fn default_temperature_scaling() -> f64 {
    1.0
}

fn default_logit_scaling() -> f64 {
    0.1
}

impl Config {
    // get config from the json file
    pub fn from_json(path: &str) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut config: Config = from_reader(reader)?;
        
        let learning = config.learning.to_ascii_lowercase();
        //determine `unsupervised` if there's label_index missing and the data is unstructured
        config.learning_task = Config::determine_learning(config.label_column_index, &config.columns, learning);
        config.sequence_data = Config::compute_sequence_data(config.sequence_data, config.learning_task);
        config.checkpoint_interval = Config::checkpoint_interval(config.epochs, config.check_points);
        if config.columns.is_none() {
            println!("Warning, no columns were defined.  Assuming fixed feature set.")
        }

        Ok(config)
    }

    pub fn compute_sequence_data(original_value: bool, learning_task:LearningTask) -> bool {
        if learning_task == LearningTask::Unsupervised {
            false  // No labels? No need for sequencing.
        } else {
            original_value  // Use the value from JSON
        }
    }

fn determine_learning(label_column_index: usize, columns: &Option<ColumnsConfig>, learning: String) -> LearningTask {
    if label_column_index == usize::MAX && columns.is_none() || learning == "unsupervised" {
        return LearningTask::Unsupervised;
    }

    match learning.as_str() {
        "classification" => LearningTask::Classification,
        "regression" => LearningTask::Regression,
        _ => {
            println!("Warning: Unknown learning type '{}', defaulting to Regression.", learning);
            LearningTask::Regression
        }
    }
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
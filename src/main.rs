
mod data_loader;
mod model;
pub mod config;
pub mod matrix;
pub mod training_logs;
pub mod activation_functions;
pub mod csv_plot;
use config::Config;
use data_loader::DataLoader;
use model::Model;
use crate::model::{TaskEnum, ClassificationTaskImpl, RegressionTaskImpl}; 
use config::LearningTask; 


static DATA_PATH: &str = "./data/{1}/{2}";

fn main() {
    let project = "fashion";
    let config_location = DATA_PATH.replace("{1}",project).replace("{2}", "config.json");
    let config = match Config::from_json(config_location.as_str()) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            return;
        }
    };

    let mut data_loader = DataLoader::new(&config);
    if let Err(e) = data_loader.load_data(&project) {
        eprintln!("Error loading data: {}", e);
        return;
    }

    let task = match config.learning_task {
        LearningTask::Classification => TaskEnum::Classification(ClassificationTaskImpl),
        LearningTask::Regression => TaskEnum::Regression(RegressionTaskImpl),
        LearningTask::Unsupervised => {
            eprintln!("Unsupervised learning is not implemented");
            return;
        }
    };

    // Create the model
    let mut model = match Model::from_json(&config, &mut data_loader, task) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error creating model: {}", e);
            return;
        }
    };

    model.print_config();
    model.train();
    // csv_plot::plot_gradients("./log_files/norms.csv", "./log_files/norms.png", true);
}



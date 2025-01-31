mod data_loader;
mod model;
pub mod config;
pub mod matrix;
pub mod activation_functions;
use config::Config;
use data_loader::DataLoader;
use model::Model;
static DATA_PATH: &str = "./data/{1}/{2}";

fn main() {
    let project = "stocks";
    let config_location = DATA_PATH.replace("{1}",project).replace("{2}", "config.json");
    let evaluation_location:String = DATA_PATH.replace("{1}",project).replace("{2}", "evaluation.json");
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

    let mut model = match Model::from_json(&config, &mut data_loader, &project) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error creating model: {}", e);
            return;
        }
    };
    model.print_config();
    model.train();
    model.evaluate(Some(&evaluation_location));
}
mod data_loader;
mod model;
pub mod config;
pub mod matrix;
pub mod activation_functions;
use config::Config;
use data_loader::DataLoader;
use model::Model;

fn main() {
    let config_location = "/Users/richarddalley/Code/Rust/Transformer/data/config.json";
    let config = match Config::from_json(config_location) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            return;
        }
    };

    let mut data_loader = DataLoader::new(&config);
    if let Err(e) = data_loader.load_data() {
        eprintln!("Error loading data: {}", e);
        return;
    }

    let mut model = match Model::from_json(&config, &mut data_loader) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error creating model: {}", e);
            return;
        }
    };
    model.print_config();
    model.train();
    model.evaluate(Some("./data/evaluation.json"));
}
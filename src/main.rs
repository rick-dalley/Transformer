mod model;
pub mod matrix;
pub mod activation_functions;
use model::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut nn = Model::from_json("/Users/richarddalley/Code/Rust/NeuralNetworkRust/data/config.json")?;
    nn.print_config();
    nn.load_data()?; // Load data from the CSV file
    nn.train(true);

    Ok(())
}

use std::error::Error;
use std::env;
use linear_regression::linear_regression::linear_regression;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: train <dataset_path> <model_path> <iteration>");
        std::process::exit(1);
    }
    let dataset_path = &args[1];
    let model_path = &args[2];
    let iteration: usize = args[3].parse()?;

    let mut dataset = linear_regression::Dataset::new();
    dataset.load(dataset_path)?;
    dataset.normalize();

    let mut model = linear_regression::LinearModel::new();
    if let Err(_) = model.load(model_path) {
        println!("No model detected, creating a new model...");
    }
    model.train(&dataset, iteration);
    model.denormalize(&dataset);
    model.save(model_path)?;
    println!("Model successfully trained with {} iteration", iteration);
    Ok(())
}

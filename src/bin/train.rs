use std::error::Error;

use clap::Parser;

use linear_regression::dataset::Dataset;
use linear_regression::linear_model::LinearModel;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    dataset_path: String,
    model_path: String,
    iteration: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut dataset = Dataset::new();
    dataset.load(&args.dataset_path)?;
    dataset.normalize();
    dbg!(&dataset);

    let mut model = LinearModel::new();
    model.train(&dataset, args.iteration);
    println!(
        "Model successfully trained with {} iteration",
        args.iteration
    );
    dbg!(&model);
    println!(
        "Model precision: {}",
        model.determination_coefficient(&dataset)
    );
    model.denormalize(&dataset);
    model.save(&args.model_path)?;
    Ok(())
}

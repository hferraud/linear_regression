use std::error::Error;

use clap::Parser;

use linear_regression::dataset::Dataset;
use linear_regression::linear_model::LinearModel;

const DEFAULT_ITERATION: usize = 100000;
const DEFAULT_LEARNING_RATE_A: f64 = 0.0000000001;
const DEFAULT_LEARNING_RATE_B: f64 = 0.001;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    dataset_path: String,
    model_path: String,

    #[arg(short, default_value_t = DEFAULT_ITERATION)]
    iteration: usize,

    #[arg(short, default_value_t = DEFAULT_LEARNING_RATE_A)]
    a_learning_rate: f64,

    #[arg(short, default_value_t = DEFAULT_LEARNING_RATE_B)]
    b_learning_rate: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut dataset = Dataset::new();
    dataset.load(&args.dataset_path)?;

    let mut model = LinearModel::new(args.a_learning_rate, args.b_learning_rate);
    model.train(&dataset, args.iteration);
    println!(
        "Model successfully trained with {} iteration",
        args.iteration
    );
    println!(
        "Model precision: {}",
        model.determination_coefficient(&dataset)
    );
    model.save(&args.model_path)?;
    Ok(())
}

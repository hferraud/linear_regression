use std::error::Error;

use clap::Parser;
use plotters::prelude::*;

use linear_regression::linear_regression;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    dataset_path: String,
    model_path: String,
    iteration: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let root = SVGBackend::new("graph.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut dataset = linear_regression::Dataset::new();
    dataset.load(&args.dataset_path)?;
    dataset.normalize();

    let mut model = linear_regression::LinearModel::new();
    model.train(&dataset, args.iteration);
    println!(
        "Model successfully trained with {} iteration",
        args.iteration
    );
    println!(
        "Model precision: {}",
        model.determination_coefficient(&dataset)
    );
    model.denormalize(&dataset);
    model.save(&args.model_path)?;
    Ok(())
}

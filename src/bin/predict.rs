use std::error::Error;

use clap::Parser;

use linear_regression::linear_regression::LinearModel;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    model_path: String,
    mileage: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut model = LinearModel::new();
    if let Err(_) = model.load(&args.model_path) {
        println!("No model detected, creating a new model...");
    }
    println!(
        "Estimated price for {} km: {}",
        args.mileage,
        model.estimate(args.mileage)
    );
    Ok(())
}

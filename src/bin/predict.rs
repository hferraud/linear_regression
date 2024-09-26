use std::error::Error;

use clap::Parser;

use linear_regression::linear_model::LinearModel;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    mileage: f64,
    model_path: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut model = LinearModel::new();
    if let Some(model_path) = args.model_path {
        model.load(&model_path)?;
    }
    println!(
        "Estimated price for {} km: {}",
        args.mileage,
        model.predict(args.mileage)
    );
    Ok(())
}

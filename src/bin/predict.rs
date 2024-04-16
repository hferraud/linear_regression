use std::error::Error;
use std::env;
use linear_regression::linear_regression::linear_regression::LinearModel;
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let model_path = &args[1];
    let mileage: f64 = args[2].parse()?;

    let mut model = LinearModel::new();
    model.load(model_path)?;
    println!("Estimated price for {} km: {}", mileage ,model.estimate(mileage));
    Ok(())
}

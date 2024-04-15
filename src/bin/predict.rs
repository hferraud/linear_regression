use serde::de::Error;
use linear_regression::linear_regression::linear_regression::LinearModel;
fn main() -> Result<(), Box<dyn Error>> {
    let model = LinearModel::load()?;
    println!("estimated price for 12000km: {}", model.estimate(61000.));
    Ok(())
}

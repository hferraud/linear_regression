use std::error::Error;
use linear_regression::linear_regression::linear_regression;

fn main() -> Result<(), Box<dyn Error>> {
    let saved_model: linear_regression::LinearModel = linear_regression::LinearModel {
        a: 2.,
        b: 3.,
        learning_rate: 0.5,
    };
    saved_model.save_linear_model()?;
    let mut dataset = linear_regression::Dataset::new();
    dataset.load("assets/data.csv")?;
    // dbg!(dataset);
    saved_model.train(&dataset)?;
    Ok(())
}

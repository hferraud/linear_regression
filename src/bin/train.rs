use std::error::Error;
use linear_regression::linear_regression::linear_regression;

fn main() -> Result<(), Box<dyn Error>> {
    let mut saved_model: linear_regression::LinearModel = linear_regression::LinearModel {
        a: -0.011,
        b: 6662.,
        learning_rate: 0.00000001,
    };
    saved_model.save()?;
    let mut dataset = linear_regression::Dataset::new();
    dataset.load("assets/data.csv")?;
    saved_model.train(&dataset, 100);
    println!("a: {}, b: {}", saved_model.a, saved_model.b);
    saved_model.save()?;
    Ok(())
}

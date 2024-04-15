use std::error::Error;
use linear_regression::linear_regression::linear_regression;

fn main() -> Result<(), Box<dyn Error>> {
    let mut saved_model: linear_regression::LinearModel = linear_regression::LinearModel {
        a: 0.,
        b: 0.,
        learning_rate: 0.1,
    };
    saved_model.save()?;
    let mut dataset = linear_regression::Dataset::new();
    dataset.load("assets/data.csv")?;
    dataset.normalize();
    saved_model.train(&dataset, 10000000);
    println!("Normalized | a: {}, b: {}", saved_model.a, saved_model.b);
    saved_model.denormalize(&dataset);
    println!("Denormalized | a: {}, b: {}", saved_model.a, saved_model.b);
    saved_model.save()?;
    Ok(())
}

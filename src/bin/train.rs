use std::error::Error;
use linear_regression::linear_regression::linear_regression;

fn main() -> Result<(), Box<dyn Error>> {
    let mut saved_model: linear_regression::LinearModel = linear_regression::LinearModel {
        a: 0.,
        b: 0.,
        learning_rate: 0.01,
    };
    saved_model.save_linear_model()?;
    let mut dataset = linear_regression::Dataset::new();
    dataset.load("assets/data.csv")?;
    // dbg!(dataset);
    saved_model.train(&dataset, 100);
    println!("a: {}, b: {}", saved_model.a, saved_model.b);
    Ok(())
}

use linear_regression::linear_regression::linear_regression::LinearModel;
fn main() {
    let model = LinearModel::load_linear_model()?;
    println!("estimated price for 12000km: {}", model.estimate(61000.))
}

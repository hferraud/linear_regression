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
    let root = BitMapBackend::new("graph.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut dataset = linear_regression::Dataset::new();
    dataset.load(&args.dataset_path)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Car price by mileage", ("sans-serif", 20).into_font())
        .x_label_area_size(20)
        .y_label_area_size(50)
        .build_cartesian_2d(
            dataset.x.min - 10000f64..dataset.x.max + 10000f64,
            dataset.y.min - 500f64..dataset.y.max + 500f64)?;
    chart
        .configure_mesh()
        .draw()?;

    dataset.draw(&mut chart)?;
    dataset.normalize();

    let mut model = linear_regression::LinearModel::new();
    if let Err(_) = model.load(&args.model_path) {
        println!("No model detected, creating a new model...");
    }
    model.train(&dataset, args.iteration);
    model.denormalize(&dataset);
    model.draw(&mut chart, &dataset)?;
    model.save(&args.model_path)?;
    println!(
        "Model successfully trained with {} iteration",
        args.iteration
    );
    Ok(())
}

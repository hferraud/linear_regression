use clap::Parser;
use plotters::prelude::*;
use std::error::Error;
use std::ops::Range;

use linear_regression::dataset::Dataset;
use linear_regression::linear_model::LinearModel;

const DEFAULT_ITERATION: usize = 100000;
const DEFAULT_LEARNING_RATE_A: f64 = 13e-11;
const DEFAULT_LEARNING_RATE_B: f64 = 0.1;
const CARTESIAN_X_RANGE: Range<f64> = 0f64..250000f64;
const CARTESIAN_Y_RANGE: Range<f64> = 0f64..9000f64;
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    dataset_path: String,
    model_path: String,

    #[arg(short, default_value_t = DEFAULT_ITERATION)]
    iteration: usize,

    #[arg(short, default_value_t = DEFAULT_LEARNING_RATE_A)]
    a_learning_rate: f64,

    #[arg(short, default_value_t = DEFAULT_LEARNING_RATE_B)]
    b_learning_rate: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut dataset = Dataset::new();
    dataset.load(&args.dataset_path)?;

    let mut model = LinearModel::new();
    model.set_learning_rate(args.a_learning_rate, args.b_learning_rate);
    model.train(&dataset, args.iteration);
    println!(
        "Model successfully trained with {} iteration",
        args.iteration
    );
    println!(
        "Model precision: {}",
        model.mean_absolute_error(&dataset)
    );
    model.save(&args.model_path)?;
    plot(&dataset, &model)?;
    Ok(())
}

fn plot(dataset: &Dataset, linear_model: &LinearModel) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("assets/plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Car price by mileage prediction",
            ("sans-serif", 30).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(CARTESIAN_X_RANGE, CARTESIAN_Y_RANGE)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(PointSeries::of_element(
        dataset
            .x
            .data
            .iter()
            .copied()
            .zip(dataset.y.data.iter().copied()),
        3,
        &RED,
        &|coord, size, style| Circle::new(coord, size, style.filled()),
    ))?;
    chart.draw_series(LineSeries::new(
        dataset.x.data.iter().filter_map(|x| {
            match linear_model.predict(*x) > CARTESIAN_X_RANGE.start
                && linear_model.predict(*x) < CARTESIAN_X_RANGE.end
            {
                true => Some((*x, linear_model.predict(*x))),
                false => None,
            }
        }),
        &BLACK,
    ))?;
    root.present()?;
    Ok(())
}

use std::error::Error;
use std::fs::File;

use serde::{Deserialize, Serialize};

use crate::dataset::Dataset;

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearModel {
    pub a: f64,
    pub b: f64,
    pub learning_rate: f64,
}

impl LinearModel {
    pub fn new() -> Self {
        LinearModel {
            a: 0.,
            b: 0.,
            learning_rate: 0.2,
        }
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::open(path)?;
        let mut reader = csv::Reader::from_reader(file);
        if let Some(result) = reader.deserialize().next() {
            *self = result?;
            Ok(())
        } else {
            Err("An error occurred while loading the model".into())
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut writer = csv::Writer::from_path(path)?;
        writer.serialize(self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn estimate(&self, x: f64) -> f64 {
        self.a * x + self.b
    }

    pub fn train(&mut self, dataset: &Dataset, size: usize) {
        for _ in 0..size {
            self.gradient_descent(dataset);
        }
    }

    fn gradient_descent(&mut self, dataset: &Dataset) {
        let tmp_a = self.a - self.learning_rate * self.cost_a(dataset);
        let tmp_b = self.b - self.learning_rate * self.cost_b(dataset);
        self.a = tmp_a;
        self.b = tmp_b;
    }

    fn cost_a(&self, dataset: &Dataset) -> f64 {
        let mut result: f64 = 0.;
        for (key, value) in dataset {
            result += (self.estimate(*key) - *value) * *key;
        }
        return result / dataset.len() as f64;
    }

    fn cost_b(&self, dataset: &Dataset) -> f64 {
        let mut result: f64 = 0.;
        for (key, value) in dataset {
            result += self.estimate(*key) - *value;
        }
        return result / dataset.len() as f64;
    }

    pub fn determination_coefficient(&self, dataset: &Dataset) -> f64 {
        let dataset_mean: f64 = dataset.y.data.iter().sum::<f64>() / dataset.y.data.len() as f64;
        let square_sum_total: f64 = dataset.y.data.iter().map(|y| (y - dataset_mean).powi(2)).sum();
        let y_pred: Vec<f64> = dataset.x.data.iter().map(|x| (self.a * x + self.b)).collect();
        let square_sum_residual: f64 = dataset.y.data.iter().zip(y_pred.iter()).map(|(y_true, y_pred)| (y_true - y_pred).powi(2)).sum();
        1.0 - (square_sum_residual / square_sum_total)
    }

    pub fn denormalize(&mut self, dataset: &Dataset) {
        let range_x = dataset.x.max - dataset.x.min;
        let range_y = dataset.y.max - dataset.y.min;
        self.a = (range_y) / (range_x) * self.a;
        self.b = range_y * self.b + dataset.y.min - range_y / range_x * dataset.x.min * self.a;
    }
}

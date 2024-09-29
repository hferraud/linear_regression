use std::error::Error;
use std::fs::File;

use serde::{Deserialize, Serialize};

use crate::dataset::Dataset;

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearModel {
    pub a: f64,
    pub b: f64,
    pub learning_rate_a: f64,
    pub learning_rate_b: f64,
}

impl LinearModel {
    pub fn new() -> Self {
        LinearModel {
            a: 0.,
            b: 0.,
            learning_rate_a: 0.,
            learning_rate_b: 0.,
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate_a: f64, learning_rate_b: f64) {
        self.learning_rate_a = learning_rate_a;
        self.learning_rate_b = learning_rate_b;
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let Ok(file) = File::open(path) else {
            return Err(format!("Couldn't open the file {path}").into());
        };
        let mut reader = csv::Reader::from_reader(file);
        let Some(result) = reader.deserialize().next() else {
            return Err("An error occurred while loading the model".into());
        };
        let Ok(model) = result else {
            return Err("An error occurred while parsing the model".into());
        };
        *self = model;
        Ok(())
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut writer = csv::Writer::from_path(path)?;
        writer.serialize(self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.a * x + self.b
    }

    pub fn train(&mut self, dataset: &Dataset, iteration: usize) {
        for _ in 0..iteration {
            self.gradient_descent(dataset);
        }
    }

    fn gradient_descent(&mut self, dataset: &Dataset) {
        let tmp_a = self.a - self.learning_rate_a * self.gradient_a(dataset);
        let tmp_b = self.b - self.learning_rate_b * self.gradient_b(dataset);
        self.a = tmp_a;
        self.b = tmp_b;
    }

    fn gradient_a(&self, dataset: &Dataset) -> f64 {
        let mut sum: f64 = 0.;
        for (x, y) in dataset {
            sum += (self.predict(*x) - y) * x;
        }
        sum / dataset.len() as f64
    }

    fn gradient_b(&self, dataset: &Dataset) -> f64 {
        let mut sum: f64 = 0.;
        for (x, y) in dataset {
            sum += self.predict(*x) - y;
        }
        sum / dataset.len() as f64
    }

    pub fn mean_absolute_error(&self, dataset: &Dataset) -> f64 {
        let squared_error: f64 = dataset.into_iter().map(|(x, y)|
                (y - self.predict(*x)).abs()
            ).sum();
        squared_error / dataset.len() as f64
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn model_load_success() {
        let mut model = LinearModel::new();
        model.load("tests/model/success").unwrap();
        assert_eq!(model.a, -0.5);
        assert_eq!(model.b, 0.5);
        assert_eq!(model.learning_rate_a, 0.1);
        assert_eq!(model.learning_rate_b, 0.2);
    }

    #[test]
    #[should_panic(expected = "Couldn't open the file")]
    fn model_load_no_file() {
        let mut model = LinearModel::new();
        model.load("tests/model/no_file").unwrap();
    }

    #[test]
    #[should_panic(expected = "An error occurred while loading the model")]
    fn model_load_empty() {
        let mut model = LinearModel::new();
        model.load("tests/model/empty").unwrap();
    }

    #[test]
    #[should_panic(expected = "An error occurred while loading the model")]
    fn model_load_no_value() {
        let mut model = LinearModel::new();
        model.load("tests/model/no_value").unwrap();
    }

    #[test]
    #[should_panic(expected = "An error occurred while parsing the model")]
    fn model_load_invalid_value() {
        let mut model = LinearModel::new();
        model.load("tests/model/invalid_value").unwrap();
    }
}

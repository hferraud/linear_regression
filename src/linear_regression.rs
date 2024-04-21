use std::error::Error;
use std::fs::File;

use csv::Writer;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

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
            learning_rate: 0.1,
        }
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::open(path)?;
        let mut reader = csv::Reader::from_reader(file);
        if let Some(result) = reader.deserialize().next() {
            let model: Self = result?;
            *self = model;
            Ok(())
        } else {
            Err("An error occurred while loading the model".into())
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut writer = Writer::from_path(path)?;
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

    pub fn denormalize(&mut self, dataset: &Dataset) {
        let range_x = dataset.x.max - dataset.x.min;
        let range_y = dataset.y.max - dataset.y.min;
        self.a = (range_y) / (range_x) * self.a;
        self.b = range_y * self.b + dataset.y.min - range_y / range_x * dataset.x.min * self.a;
    }

    pub fn draw(&self, chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>, dataset: &Dataset) -> Result<(), Box<dyn Error>> {
        let x_start = dataset.x.min;
        let x_end = dataset.x.max;
        let y_start = self.a * x_start + self.b;
        let y_end = self.a * x_end + self.b;
        chart.draw_series(LineSeries::new(vec![(x_start, y_start), (x_end, y_end)], &BLUE))?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct DatasetRow {
    data: Vec<f64>,
    pub min: f64,
    pub max: f64,
}

impl DatasetRow {
    pub fn new() -> Self {
        DatasetRow {
            data: Vec::new(),
            min: 0.,
            max: 0.,
        }
    }

    pub fn push(&mut self, data: f64) {
        self.data.push(data);
    }

    pub fn len(&self) -> usize {
        return self.data.len();
    }

    pub fn set_range(&mut self) {
        self.min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        self.max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }

    fn normalize(&mut self) {
        let range = self.max - self.min;

        for value in self.data.iter_mut() {
            *value = (*value - self.min) / range;
        }
    }
}

#[derive(Debug)]
pub struct Dataset {
    pub x: DatasetRow,
    pub y: DatasetRow,
}

impl Dataset {
    pub fn new() -> Self {
        Dataset {
            x: DatasetRow::new(),
            y: DatasetRow::new(),
        }
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::open(path)?;
        let mut reader = csv::Reader::from_reader(file);
        for result in reader.deserialize() {
            let record: (f64, f64) = result?;
            self.push(record);
        }
        self.y.set_range();
        self.x.set_range();
        Ok(())
    }

    pub fn push(&mut self, row: (f64, f64)) {
        self.x.push(row.0);
        self.y.push(row.1);
    }

    pub fn len(&self) -> usize {
        return self.x.len();
    }

    pub fn draw(&self, chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), Box<dyn Error>> {
        chart.draw_series(PointSeries::of_element(
            self.x.data.iter().zip(self.y.data.iter()).map(|(&x, &y)| (x, y)),
            2,
            &RED,
            &|c, s, st: ShapeStyle| {
                return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
            },
        ))?;
        Ok(())
    }

    pub fn normalize(&mut self) {
        self.x.normalize();
        self.y.normalize();
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = (&'a f64, &'a f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let keys_ref: &'a Vec<f64> = &self.x.data;
        let values_ref: &'a Vec<f64> = &self.y.data;
        let tuples = keys_ref.iter().zip(values_ref.iter());
        tuples.collect::<Vec<_>>().into_iter()
    }
}

impl IntoIterator for Dataset {
    type Item = (f64, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let keys = self.x.data;
        let values = self.y.data;
        let tuples = keys.into_iter().zip(values.into_iter());
        tuples.collect::<Vec<_>>().into_iter()
    }
}

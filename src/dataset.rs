use std::error::Error;
use std::fs::File;

#[derive(Debug)]
pub struct DatasetRow {
    pub data: Vec<f64>,
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
            // let Ok(record) = result else {
            //     return Err("An error occurred while loading the model".into());
            // };
            self.push(result?);
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

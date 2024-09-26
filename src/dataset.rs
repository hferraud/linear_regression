use std::error::Error;
use std::fs::File;

#[derive(Debug)]
pub struct DatasetRow {
    pub data: Vec<f64>,
}

impl DatasetRow {
    pub fn new() -> Self {
        DatasetRow { data: Vec::new() }
    }

    pub fn push(&mut self, data: f64) {
        self.data.push(data);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug)]
pub struct Dataset {
    pub x: DatasetRow,
    pub y: DatasetRow,
}

impl Dataset {
    const OPEN_FILE_ERROR: &'static str = "Couldn't open the file";
    const LOAD_ERROR: &'static str = "An error occurred while loading the dataset";
    const NB_ROW_ERROR: &'static str = "The dataset should contain at least 2 row";

    pub fn new() -> Self {
        Dataset {
            x: DatasetRow::new(),
            y: DatasetRow::new(),
        }
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let Ok(file) = File::open(path) else {
            return Err(Self::OPEN_FILE_ERROR.into());
        };
        let mut reader = csv::Reader::from_reader(file);
        for result in reader.deserialize() {
            let Ok(row) = result else {
                return Err(Self::LOAD_ERROR.into());
            };
            self.push(row);
        }
        self.dedup();
        if self.len() <= 1 {
            return Err(Self::NB_ROW_ERROR.into());
        }
        Ok(())
    }

    pub fn push(&mut self, row: (f64, f64)) {
        self.x.push(row.0);
        self.y.push(row.1);
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn dedup(&mut self) {
        let mut tuples: Vec<_> = self.into_iter().map(|(x, y)| (*x, *y)).collect();
        tuples.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });
        tuples.dedup();
        self.x.data = tuples.iter().map(|(x, _)| x).cloned().collect();
        self.y.data = tuples.iter().map(|(_, y)| y).cloned().collect();
    }
}

impl IntoIterator for Dataset {
    type Item = (f64, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let keys = self.x.data.clone();
        let values = self.y.data.clone();
        let tuples = keys.into_iter().zip(values.into_iter());
        tuples.collect::<Vec<_>>().into_iter()
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = (&'a f64, &'a f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let keys: &'a Vec<f64> = &self.x.data;
        let values: &'a Vec<f64> = &self.y.data;
        let tuples = keys.iter().zip(values.iter());
        tuples.collect::<Vec<_>>().into_iter()
    }
}

impl<'a> IntoIterator for &'a mut Dataset {
    type Item = (&'a mut f64, &'a mut f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let keys: &'a mut Vec<f64> = &mut self.x.data;
        let values: &'a mut Vec<f64> = &mut self.y.data;
        let tuples = keys.iter_mut().zip(values.iter_mut());
        tuples.collect::<Vec<_>>().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_load_success() {
        let mut dataset = Dataset::new();
        dataset.load("tests/dataset/success").unwrap();
    }
    #[test]
    #[should_panic(expected = "Couldn't open the file")]
    fn dataset_load_no_file() {
        let mut dataset = Dataset::new();
        dataset.load("tests/dataset/no_file").unwrap();
    }

    #[test]
    #[should_panic(expected = "The dataset should contain at least 2 row")]
    fn dataset_load_empty() {
        let mut dataset = Dataset::new();
        dataset.load("tests/dataset/empty").unwrap();
    }

    #[test]
    #[should_panic(expected = "The dataset should contain at least 2 row")]
    fn dataset_load_no_value() {
        let mut dataset = Dataset::new();
        dataset.load("tests/dataset/no_value").unwrap();
    }

    #[test]
    #[should_panic(expected = "An error occurred while loading the dataset")]
    fn dataset_load_invalid_value() {
        let mut dataset = Dataset::new();
        dataset.load("tests/model/invalid_value").unwrap();
    }

    #[test]
    #[should_panic(expected = "The dataset should contain at least 2 row")]
    fn dataset_one_row() {
        let mut dataset = Dataset::new();
        dataset.load("tests/dataset/one_row").unwrap();
    }

    #[test]
    fn dataset_dedup() {
        let mut dataset = Dataset::new();
        dataset.load("tests/dataset/dedup").unwrap();
        assert_eq!(dataset.len(), 4);
    }
}

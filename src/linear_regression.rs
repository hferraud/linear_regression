pub mod linear_regression {
    use std::error::Error;
    use std::fs::File;
    use csv::Writer;
    use serde::{Deserialize, Serialize};

    const MODEL_PATH: &str = "assets/linear_model.csv";

    #[derive(Debug, Serialize, Deserialize)]
    pub struct LinearModel {
        pub a: f64,
        pub b: f64,
        pub learning_rate: f64,
    }

    impl LinearModel {
        pub fn estimate(&self, x: f64) -> f64 {
            self.a * x + self.b
        }

        pub fn load_linear_model() -> Result<Self, Box<dyn Error>> {
            let file = File::open(MODEL_PATH)?;
            let mut reader = csv::Reader::from_reader(file);
            if let Some(result) = reader.deserialize().next() {
                let model: Self = result?;
                Ok(model)
            } else {
                return Err("test".into())
            }
        }

        pub fn save_linear_model(&self) -> Result<(), Box<dyn Error>> {
            let mut writer = Writer::from_path(MODEL_PATH)?;
            writer.serialize(self)?;
            writer.flush()?;
            Ok(())
        }

        pub fn train(&mut self, dataset: &Dataset, size: usize) {
            for _ in 0..size {
                self.gradient_descent(dataset);
            }
        }

        fn gradient_descent(&mut self, dataset: &Dataset) {
            self.a -= self.learning_rate * self.cost_a(dataset);
            self.b -= self.learning_rate * self.cost_b(dataset);
        }

        fn cost_a(&self, dataset: &Dataset) -> f64 {
            let mut result: f64 = 0.;
            for (key, value) in dataset {
                result += self.estimate(*key) - *value * *key;
                println!("result: {result}");
                println!("key: {}", *key);
                println!("value: {}", *value);
                println!("estimate: {}", self.estimate(*key));
                println!();
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
    }

    #[derive(Debug)]
    pub struct Dataset {
        key: Vec<f64>,
        value: Vec<f64>,
    }

    impl Dataset {
        pub fn new() -> Self {
            Dataset {
                key: Vec::new(),
                value: Vec::new(),
            }
        }

        pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
            let file = File::open(path)?;
            let mut reader = csv::Reader::from_reader(file);
            for result in reader.deserialize() {
                let record: (f64, f64) = result?;
                self.push(record);
            }
            Ok(())
        }

        pub fn push(&mut self, row: (f64, f64)) {
            self.key.push(row.0);
            self.value.push(row.1);
        }

        pub fn len(&self) -> usize {
            return self.key.len();
        }
    }

    impl<'a> IntoIterator for &'a Dataset {
        type Item = (&'a f64, &'a f64);
        type IntoIter = std::vec::IntoIter<Self::Item>;

        fn into_iter(self) -> Self::IntoIter {
            let keys_ref: &'a Vec<f64> = &self.key;
            let values_ref: &'a Vec<f64> = &self.value;
            let tuples = keys_ref.iter().zip(values_ref.iter());
            tuples.collect::<Vec<_>>().into_iter()
        }
    }


    impl IntoIterator for Dataset {
        type Item = (f64, f64);
        type IntoIter = std::vec::IntoIter<Self::Item>;

        fn into_iter(self) -> Self::IntoIter {
            let keys = self.key;
            let values = self.value;
            let tuples = keys.into_iter().zip(values.into_iter());
            tuples.collect::<Vec<_>>().into_iter()
        }
    }


}
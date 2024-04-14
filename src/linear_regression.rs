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

        pub fn train(&self, dataset: &Dataset) -> Result<(), Box<dyn Error>> {
            for (key, value) in dataset {
                println!("{:?}, {:?}", key, value);
            }
            Ok(())
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


    pub fn estimate(x: f32, a: f32, b: f32) -> f32 {
        a * x + b
    }

}
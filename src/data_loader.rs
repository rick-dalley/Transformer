use crate::config;
use redis::{Commands, RedisResult};
use postgres::{Client, NoTls};
use csv::ReaderBuilder;
use std::io::Write;
use rand::prelude::SliceRandom;
use crate::matrix::Matrix;

pub struct DataLoader {
    pub data_source: String,
    pub data_location: String,
    pub connection_string: Option<String>,
    pub cap_data_rows: bool,
    pub max_data_rows: usize,
    pub columns: config::ColumnsConfig,
    pub sequence_length: usize,
    pub training_data:Matrix,
    pub training_labels: Vec<usize>,
    pub validation_data:Matrix,
    pub validation_labels: Vec<usize>,
    pub validation_split: f64,
    pub split_index:usize,
}

impl DataLoader {

    pub fn new(config: &config::Config) -> Self {
        let split_index = ((1.0 - config.validation_split) * config.sequence_length as f64) as usize;
        let data_source= config.data_source.clone();
        let data_location= config.location.clone();
        let connection_string= Some(config.connection_string.clone());
        let cap_data_rows=config.cap_data_rows;
        let max_data_rows=config.max_data_rows;
        let sequence_length=config.sequence_length;
        let columns=config.columns.clone();
        let training_data= Matrix::zeros(0, config.sequence_length);
        let training_labels= vec![];
        let validation_data= Matrix::zeros(0, config.sequence_length);
        let validation_labels= vec![];
        let validation_split=config.validation_split;
        Self {
            data_source,
            data_location,
            connection_string,
            cap_data_rows,
            max_data_rows,
            columns,
            sequence_length,
            training_data,
            training_labels,
            validation_data,
            validation_labels,
            validation_split,
            split_index,
        }
        
    }


    pub fn load_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.data_source.as_str() {
            "file" => self.load_from_file(),
            "redis" | "postgres" => self.load_from_db(),
            _ => Err(format!("Unsupported data source: {}", self.data_source).into()),
        }
    }

    pub fn load_from_db(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.data_source.as_str() {
            "redis" => self.load_from_redis(),
            "postgres" => self.load_from_postgres(),
            _ => Err(format!("Unsupported database type: {}", self.data_source).into()),
        }
    }

    // Load data from Redis
    fn load_from_redis(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let connection_str = self.connection_string
            .as_deref() // Converts `Option<String>` into `Option<&str>`
            .ok_or("Missing connection string for Redis")?;

        let client = redis::Client::open(connection_str)?;

        let mut con = client.get_connection()?;

        let keys: RedisResult<Vec<String>> = con.keys("*");
        if keys.is_err() {
            return Err("No keys found in Redis.".into());
        }

        let mut raw_data: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        let mut categorical_values: Vec<String> = Vec::new();
        
        let mut row_count = 0;
        let mut skipped_rows = 0;

        for key in keys.unwrap() {
            if self.cap_data_rows && row_count >= self.max_data_rows {
                break;
            }
            let value: String = con.get(&key)?;
            let record: Vec<String> = serde_json::from_str(&value)?;

            let (valid, features, label, category) = self.process_record(&record);
            
            if valid {
                raw_data.push(features);
                labels.push(label);
                categorical_values.push(category);
                row_count += 1;
            } else {
                skipped_rows += 1;
            }
        }

        println!(
            "Loaded {} rows from Redis. Skipped {} invalid rows.",
            row_count, skipped_rows
        );

        self.process_loaded_data(raw_data, labels, categorical_values)
    }

    // Load data from PostgreSQL
    fn load_from_postgres(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let connection_str = self.connection_string
            .as_deref() // Converts `Option<String>` into `Option<&str>`
            .ok_or("Missing connection string for Redis")?;

        let mut client = Client::connect(connection_str, NoTls)?;

        let feature_columns = self.columns.features.join(", ");
        let target_column = &self.columns.target;
        let categorical_column = &self.columns.categorical_column;

        let query = format!(
            "SELECT {}, {}, {} FROM my_table",
            feature_columns, target_column, categorical_column
        );

        let rows = client.query(query.as_str(), &[])?;

        let mut raw_data: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        let mut categorical_values: Vec<String> = Vec::new();

        let mut row_count = 0;
        let mut skipped_rows = 0;

        for row in rows {
            if self.cap_data_rows && row_count >= self.max_data_rows {
                break;
            }

            let record: Vec<String> = (0..self.columns.features.len() + 2)
                .map(|i| row.get::<_, String>(i))
                .collect();

            let (valid, features, label, category) = self.process_record(&record);

            if valid {
                raw_data.push(features);
                labels.push(label);
                categorical_values.push(category);
                row_count += 1;
            } else {
                skipped_rows += 1;
            }
        }

        println!(
            "Loaded {} rows from PostgreSQL. Skipped {} invalid rows.",
            row_count, skipped_rows
        );

        self.process_loaded_data(raw_data, labels, categorical_values)
    }

    // Process a single record (for both Redis and PostgreSQL)
    fn process_record(
        &self,
        record: &[String]
    ) -> (bool, Vec<f64>, f64, String) {
        let mut valid = true;
        let mut features = Vec::new();
        let mut errors = Vec::new();

        for (i, value) in record.iter().enumerate() {
            if i < self.columns.features.len() {
                match value.parse::<f64>() {
                    Ok(num) => features.push(num),
                    Err(_) => {
                        valid = false;
                        errors.push(format!("Invalid numeric value in column {}", i));
                    }
                }
            }
        }

        let target_value = record[self.columns.features.len()].parse::<f64>().unwrap_or_else(|_| {
            valid = false;
            errors.push("Invalid target value".to_string());
            0.0
        });

        let category_value = record[self.columns.features.len() + 1].clone();

        if category_value.is_empty() {
            valid = false;
            errors.push("Empty value in categorical column".to_string());
        }

        if !valid {
            println!("Skipping row due to errors: {:?}", errors);
        }

        (valid, features, target_value, category_value)
    }

    // Final processing of loaded data (shared for file, Redis, and Postgres)
    fn process_loaded_data(
        &mut self,
        mut raw_data: Vec<Vec<f64>>,
        labels: Vec<f64>,
        categorical_values: Vec<String>
    ) -> Result<(), Box<dyn std::error::Error>> {
        if raw_data.is_empty() {
            return Err("No valid data to process".into());
        }

        let num_features = raw_data[0].len();
        let mut feature_means = vec![0.0; num_features];
        let mut feature_stds = vec![0.0; num_features];

        for row in &raw_data {
            for (i, &value) in row.iter().enumerate() {
                feature_means[i] += value;
            }
        }
        feature_means.iter_mut().for_each(|mean| *mean /= raw_data.len() as f64);

        for row in &raw_data {
            for (i, &value) in row.iter().enumerate() {
                feature_stds[i] += (value - feature_means[i]).powi(2);
            }
        }
        feature_stds.iter_mut().for_each(|std| *std = (*std / raw_data.len() as f64).sqrt());

        for row in &mut raw_data {
            for (i, value) in row.iter_mut().enumerate() {
                *value = (*value - feature_means[i]) / feature_stds[i];
            }
        }

        let categorical_map: std::collections::HashMap<String, usize> = categorical_values
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, value)| (value, idx))
            .collect();
        let categorical_indices: Vec<usize> = categorical_values
            .iter()
            .map(|value| *categorical_map.get(value).unwrap())
            .collect();

        let mut data: Vec<Vec<f64>> = Vec::new();
        let mut sequence_labels: Vec<f64> = Vec::new();

        for i in 0..(raw_data.len() - self.sequence_length) {
            let mut sequence: Vec<f64> = Vec::new();

            for j in 0..self.sequence_length {
                sequence.extend(&raw_data[i + j]);
                sequence.push(categorical_indices[i + j] as f64);
            }

            data.push(sequence);
            sequence_labels.push(labels[i + self.sequence_length - 1]);
        }

        let raw_data = Matrix::new(data.len(), data[0].len(), data.into_iter().flatten().collect());
        let raw_labels:Vec<usize> = sequence_labels.iter().map(|&x| x as usize).collect();

        if self.validation_split > 0.0 {
            self.split_data(&raw_data, &raw_labels);
        } else {
            self.split_index = raw_data.rows;
            self.training_data = raw_data.clone();
            self.training_labels = raw_labels.clone();
        }

        Ok(())
    }

    // load from file
    pub fn load_from_file(&mut self) -> Result<(), Box<dyn std::error::Error>> {

        
        let mut reader = ReaderBuilder::new()
            .has_headers(true) // Assume headers are present for column names
            .from_path(&self.data_location)?;

        // Extract the feature and target columns
        let feature_indices: Vec<usize> = reader
            .headers()?
            .iter()
            .enumerate()
            .filter(|(_, name)| self.columns.features.contains(&name.to_string()))
            .map(|(idx, _)| idx)
            .collect();

        let target_index = reader
            .headers()?
            .iter()
            .position(|name| name == &self.columns.target)
            .ok_or("Target column not found in the data file")?;

        let categorical_index = reader
            .headers()?
            .iter()
            .position(|name| name == &self.columns.categorical_column)
            .ok_or("Categorical column not found in the data file")?;

        // Initialize data storage
        let mut raw_data: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        let mut categorical_values: Vec<String> = Vec::new();

        let mut row_count = 0; // Track the number of rows processed
        let mut skipped_rows = 0; // Track skipped rows

        // Open a log file for errors
        let mut error_log = std::fs::File::create("./data/error_log.csv")?;
        writeln!(error_log, "Row,Data,Error")?;

        // Read and process the data
        for record in reader.records() {
            if self.cap_data_rows && row_count >= self.max_data_rows {
                break; // Stop processing if cap_data_rows is enabled and max_data_rows is reached
            }

            let record = record?;
            row_count += 1;

            // Check for missing or invalid values
            let mut valid = true;
            let mut errors = Vec::new();

            // Validate feature columns
            let features: Vec<f64> = feature_indices
                .iter()
                .map(|&idx| {
                    record[idx]
                        .parse::<f64>()
                        .map_err(|_| format!("Missing or invalid value in feature column {}", idx))
                })
                .filter_map(|res| match res {
                    Ok(val) => Some(val),
                    Err(e) => {
                        valid = false;
                        errors.push(e);
                        None
                    }
                })
                .collect();

            // Validate target column
            let target = record[target_index].parse::<f64>().map_err(|_| {
                valid = false;
                format!("Missing or invalid value in target column {}", target_index)
            });

            if target.is_err() {
                errors.push(target.unwrap_err());
            } else {
                labels.push(target.unwrap());
            }

            // Validate categorical column
            if record[categorical_index].is_empty() {
                valid = false;
                errors.push(format!(
                    "Missing or invalid value in categorical column {}",
                    categorical_index
                ));
            } else {
                categorical_values.push(record[categorical_index].to_string());
            }

            // Log and skip invalid rows
            if !valid {
                skipped_rows += 1;
                writeln!(
                    error_log,
                    "{},{:?},{}",
                    row_count,
                    record,
                    errors.join("; ")
                )?;
                continue;
            }

            // Add valid features to raw data
            raw_data.push(features);
        }

        println!(
            "Processed {} rows. Skipped {} invalid rows.",
            row_count, skipped_rows
        );

        if raw_data.is_empty() {
            return Err("No valid data to process".into());
        }

        // Normalize features dynamically (z-score normalization)
        let num_features = raw_data[0].len();
        let mut feature_means = vec![0.0; num_features];
        let mut feature_stds = vec![0.0; num_features];

        // Calculate mean and standard deviation for each feature
        for row in &raw_data {
            for (i, &value) in row.iter().enumerate() {
                feature_means[i] += value;
            }
        }
        feature_means.iter_mut().for_each(|mean| *mean /= raw_data.len() as f64);

        for row in &raw_data {
            for (i, &value) in row.iter().enumerate() {
                feature_stds[i] += (value - feature_means[i]).powi(2);
            }
        }
        feature_stds.iter_mut().for_each(|std| *std = (*std / raw_data.len() as f64).sqrt());

        // Apply z-score normalization
        for row in &mut raw_data {
            for (i, value) in row.iter_mut().enumerate() {
                *value = (*value - feature_means[i]) / feature_stds[i];
            }
        }

        // Handle categorical column (e.g., embedding or one-hot encoding)
        let categorical_map: std::collections::HashMap<String, usize> = categorical_values
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, value)| (value, idx))
            .collect();
        let categorical_indices: Vec<usize> = categorical_values
            .iter()
            .map(|value| *categorical_map.get(value).unwrap())
            .collect();

        // Create sequences
        let mut data: Vec<Vec<f64>> = Vec::new();
        let mut sequence_labels: Vec<f64> = Vec::new();

        for i in 0..(raw_data.len() - self.sequence_length) {
            let mut sequence: Vec<f64> = Vec::new();

            for j in 0..self.sequence_length {
                sequence.extend(&raw_data[i + j]); // Add features
                sequence.push(categorical_indices[i + j] as f64); // Add categorical index
            }

            data.push(sequence);
            sequence_labels.push(labels[i + self.sequence_length - 1]); // Use last value in sequence
        }

        // Convert data to Matrix format
        let raw_data = Matrix::new(data.len(), data[0].len(), data.into_iter().flatten().collect());
        let raw_labels:Vec<usize> = sequence_labels.iter().map(|&x| x as usize).collect();

        if self.validation_split > 0.0 {
            self.split_data(&raw_data, &raw_labels);
        } else {
            // No split, assign all data to training
            self.split_index = raw_data.rows;
            self.training_data = raw_data.clone();
            self.training_labels = raw_labels.clone();
        }

        Ok(())
    }

    pub fn split_data(&mut self, data:&Matrix, labels:&Vec<usize>) {

        println!("Splitting data...");

        // Dynamically calculate split index based on validation_split
        self.split_index = ((1.0 - self.validation_split) * data.rows as f64) as usize;

        // Ensure split_index is valid
        assert!(
            self.split_index > 0 && self.split_index < data.rows,
            "Invalid split_index: {}. Ensure validation_split is correctly set.",
            self.split_index
        );

        // Split data into training and validation sets
        let data_cols = data.cols;

        // Extract rows for training data
        self.training_data = Matrix::new(
            self.split_index,
            data_cols,
            data.data[..(self.split_index * data_cols)].to_vec(),
        );

        // Extract rows for validation data
        self.validation_data = Matrix::new(
            data.rows - self.split_index,
            data_cols,
            data.data[(self.split_index * data_cols)..].to_vec(),
        );

        // Split labels into training and validation sets
        self.training_labels = labels[..self.split_index].to_vec();
        self.validation_labels = labels[self.split_index..].to_vec();

        println!(
            "Data split into training ({}) and validation ({}) sets.",
            self.training_data.rows, self.validation_data.rows
        );
    }


    pub fn shuffle_data(data: &mut Matrix, labels: &mut Vec<usize>) {
        println!("Shuffling data...");

        // Verify input alignment
        assert_eq!(
            data.rows,
            labels.len(),
            "Mismatch: data rows ({}) != labels length ({})",
            data.rows,
            labels.len()
        );

        // Generate shuffled indices
        let mut indices: Vec<usize> = (0..data.rows).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        // Create new shuffled matrices and labels
        let mut shuffled_data = Matrix::zeros(data.rows, data.cols);
        let mut shuffled_labels = Vec::with_capacity(labels.len());

        for (new_idx, &original_idx) in indices.iter().enumerate() {
            // Copy row-by-row from the original data matrix
            for col in 0..data.cols {
                shuffled_data.data[new_idx * data.cols + col] =
                    data.data[original_idx * data.cols + col];
            }

            // Copy corresponding label
            shuffled_labels.push(labels[original_idx]);
        }

        // Debug shuffled data and labels
        println!(
            "Before shuffling: First 5 labels = {:?}",
            &labels[..5]
        );
        println!(
            "After shuffling: First 5 labels = {:?}",
            &shuffled_labels[..5]
        );

        // Update the data matrix and labels
        *data = shuffled_data;
        *labels = shuffled_labels;

        println!("Shuffling complete.");
    }

}


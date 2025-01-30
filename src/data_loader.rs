use crate::config;
use redis::{Commands, RedisResult};
use postgres::{Client, NoTls};
use csv::ReaderBuilder;
use rand::prelude::SliceRandom;
use crate::matrix::Matrix;

pub struct DataLoader {
    pub data_source: String,
    pub data_location: String,
    pub connection_string: Option<String>,
    pub cap_data_rows: bool,
    pub max_data_rows: usize,
    pub sequence_length: usize,
    pub training_data:Matrix,
    pub training_labels: Vec<usize>,
    pub validation_data:Matrix,
    pub validation_labels: Vec<usize>,
    pub validation_split: f64,
    pub split_index:usize,
    pub columns: Option<config::ColumnsConfig>,
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
        let columns=config.columns.clone(); //allow 'None' if missing
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


    pub fn load_data(&mut self, project_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let error_log_location = format!("./data/{}/error_log.csv", project_name);
        match self.data_source.as_str() {
            "file" => self.load_from_file(&error_log_location),
            "redis" | "postgres" => self.load_from_db(&error_log_location),
            _ => Err(format!("Unsupported data source: {}", self.data_source).into()),
        }
    }

    pub fn load_from_db(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {
        match self.data_source.as_str() {
            "redis" => self.load_from_redis(error_log_location),
            "postgres" => self.load_from_postgres(error_log_location),
            _ => Err(format!("Unsupported database type: {}", self.data_source).into()),
        }
    }

    // Load data from Redis
    fn load_from_redis(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {

        println!("{}", error_log_location);

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
    fn load_from_postgres(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {


        println!("{}", error_log_location);

        // ✅ Early return with an error if `columns` is missing
        let columns = match self.columns.as_ref() {
            Some(columns) => columns,
            None => return Err("Error: Columns are required for structured datasets in PostgreSQL.".into()),
        };

        let connection_str = self.connection_string
            .as_deref() // Converts `Option<String>` into `Option<&str>`
            .ok_or("Missing connection string for Redis")?;

        let mut client = Client::connect(connection_str, NoTls)?;

        let feature_columns = columns.features.join(", ");
        let target_column = &columns.target;
        let categorical_column = &columns.categorical_column;

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

            let record: Vec<String> = (0..columns.features.len() + 2)
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

        // ✅ Check if columns exist before using them
        let feature_len = match &self.columns {
            Some(columns) => columns.features.len(),
            None => record.len().saturating_sub(2), // Assume all except last 2 columns are features
        };

        for (i, value) in record.iter().enumerate() {
            if i < feature_len {
                match value.parse::<f64>() {
                    Ok(num) => features.push(num),
                    Err(_) => {
                        valid = false;
                        errors.push(format!("Invalid numeric value in column {}", i));
                    }
                }
            }
        }

        // ✅ Extract target value safely
        let target_value = match record.get(feature_len) {
            Some(value) => value.parse::<f64>().unwrap_or_else(|_| {
                valid = false;
                errors.push("Invalid target value".to_string());
                0.0
            }),
            None => {
                valid = false;
                errors.push("Missing target value".to_string());
                0.0
            }
        };

        // ✅ Extract categorical value safely
        let category_value = record.get(feature_len + 1).cloned().unwrap_or_else(|| {
            valid = false;
            errors.push("Missing categorical column value".to_string());
            String::new()
        });

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


    pub fn load_from_file(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", error_log_location);

        // ✅ Extract columns before calling `self.load_for_columns`
        let columns = self.columns.clone();

        if let Some(columns) = columns {
            // ✅ Pass `columns`, avoiding borrowing `self` mutably while it's already borrowed immutably
            self.load_for_columns(&columns)
        } else {
            // ✅ Call `load_for_fixed()` safely
            self.load_for_fixed()
        }
    }

// ✅ Function to load structured datasets (CSV with defined columns)
fn load_for_columns(&mut self, columns: &config::ColumnsConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true) // Assume headers are present
        .from_path(&self.data_location)?;

    let headers = reader.headers()?.clone();

    let feature_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, name)| columns.features.contains(&name.to_string()))
        .map(|(idx, _)| idx)
        .collect();

    let target_index = headers.iter()
        .position(|name| name == &columns.target)
        .ok_or("Target column not found in the data file")?;

    let categorical_index = headers.iter()
        .position(|name| name == &columns.categorical_column)
        .ok_or("Categorical column not found in the data file")?;

    let mut raw_data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    let mut categorical_values: Vec<String> = Vec::new();

    let mut row_count = 0;
    let mut skipped_rows = 0;

    for record in reader.records() {
        let record = record?;
        row_count += 1;

        if self.cap_data_rows && row_count > self.max_data_rows {
            break;
        }

        let mut valid = true;
        let mut errors = Vec::new();

        let features: Vec<f64> = feature_indices.iter()
            .map(|&idx| record[idx].parse::<f64>().map_err(|_| format!("Invalid feature value in column {}", idx)))
            .filter_map(|res| match res {
                Ok(val) => Some(val),
                Err(e) => {
                    valid = false;
                    errors.push(e);
                    None
                }
            })
            .collect();

        let label = record[target_index].parse::<f64>().unwrap_or_else(|_| {
            valid = false;
            errors.push(format!("Invalid target value in column {}", target_index));
            0.0
        });

        categorical_values.push(record[categorical_index].to_string());

        if !valid {
            skipped_rows += 1;
            println!("Skipping row {} due to errors: {:?}", row_count, errors);
            continue;
        }

        raw_data.push(features);
        labels.push(label);
    }

    println!(
        "Processed {} rows. Skipped {} invalid rows.",
        row_count, skipped_rows
    );

    self.process_loaded_data(raw_data, labels, categorical_values)
}

// ✅ Function to load fixed-format datasets (like MNIST)
fn load_for_fixed(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    println!("Warning: No column names defined. Assuming first column is the label.");

    let mut reader = ReaderBuilder::new()
        .has_headers(false) // Assume no headers in fixed datasets
        .from_path(&self.data_location)?;

    let mut raw_data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    let mut row_count = 0;
    let mut skipped_rows = 0;

    for record in reader.records() {
        let record = record?;
        row_count += 1;

        if self.cap_data_rows && row_count > self.max_data_rows {
            break;
        }

        let mut valid = true;
        let mut errors = Vec::new();

        // ✅ First column is the label
        let label = record[0].parse::<f64>().unwrap_or_else(|_| {
            valid = false;
            errors.push("Invalid label value".to_string());
            0.0
        });

        // ✅ Remaining columns are features
        let features: Vec<f64> = record.iter()
            .skip(1) // Skip the first column (label)
            .map(|pixel| pixel.parse::<f64>().unwrap_or_else(|_| {
                valid = false;
                errors.push("Invalid pixel value".to_string());
                0.0
            }) / 255.0) // Normalize grayscale
            .collect();

        if !valid {
            skipped_rows += 1;
            println!("Skipping row {} due to errors: {:?}", row_count, errors);
            continue;
        }

        raw_data.push(features);
        labels.push(label);
    }

    println!(
        "Processed {} rows. Skipped {} invalid rows.",
        row_count, skipped_rows
    );

    self.process_loaded_data(raw_data, labels, vec![])
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


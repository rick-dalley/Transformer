use crate::config;
use redis::{Commands, RedisResult};
use postgres::{Client, NoTls};
use csv::ReaderBuilder;
// use rand::prelude::SliceRandom;
use crate::matrix::Matrix;
use rand::seq::SliceRandom;


#[derive(Debug)]
pub enum Labels {
    Classification(Vec<usize>),
    Regression(Vec<f64>),
}

impl Labels {
    // Get a mutable reference to Classification labels

    pub fn slice(&self, start: usize, end: usize) -> Labels {
        match self {
            Labels::Classification(vec) => Labels::Classification(vec[start..end.min(vec.len())].to_vec()),
            Labels::Regression(vec) => Labels::Regression(vec[start..end.min(vec.len())].to_vec()),
        }
    }

    // Get the length of the labels vector
    pub fn len(&self) -> usize {
        match self {
            Labels::Classification(vec) => vec.len(),
            Labels::Regression(vec) => vec.len(),
        }
    }

    /// Returns a reference to the regression vector or an empty slice if it's Classification.
    #[allow(dead_code)]
    pub fn as_vec_f64(&self) -> Vec<f64> {
        match self {
            Labels::Regression(vec) => vec.clone(),
            Labels::Classification(vec) => vec.iter().map(|&v| v as f64).collect(),
        }
    }

    // Shuffle the labels in place using a given index map
    pub fn shuffle_with_indices(&mut self, indices: &[usize]) {
        match self {
            Labels::Classification(vec) => {
                let mut shuffled = vec.clone();
                for (i, &idx) in indices.iter().enumerate() {
                    shuffled[i] = vec[idx];
                }
                *vec = shuffled;
            }
            Labels::Regression(vec) => {
                let mut shuffled = vec.clone();
                for (i, &idx) in indices.iter().enumerate() {
                    shuffled[i] = vec[idx];
                }
                *vec = shuffled;
            }
        }
    }
}


pub struct DataLoader {
    pub data_source: String,
    pub data_location: String,
    pub connection_string: Option<String>,
    pub cap_data_rows: bool,
    pub max_data_rows: usize,
    pub sequence_data: bool,
    pub learning_task: config::LearningTask, 
    pub sequence_length: usize,
    pub label_index: usize,
    pub training_data:Matrix,
    pub validation_data:Matrix,
    pub training_labels:  Labels,
    pub validation_labels: Labels,
    pub validation_split: f64,
    pub split_index:usize,
    pub scaling_factor:f64,
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
        let sequence_data = config.sequence_data;
        let sequence_length=config.sequence_length;
        let label_index=config.label_column_index;
        let columns=config.columns.clone(); //allow 'None' if missing
        let training_data= Matrix::zeros(0, config.sequence_length);
        let validation_data= Matrix::zeros(0, config.sequence_length);
        let training_labels = match config.learning_task {
            config::LearningTask::Classification => Labels::Classification(vec![]),
            config::LearningTask::Regression => Labels::Regression(vec![]),
            _ => panic!("Unsupported learning task"),
        };

        let validation_labels = match config.learning_task {
            config::LearningTask::Classification => Labels::Classification(vec![]),
            config::LearningTask::Regression => Labels::Regression(vec![]),
            _ => panic!("Unsupported learning task"),
        };
        let validation_split=config.validation_split;
        let learning_task = config.learning_task;
        let scaling_factor = config.scaling_factor;
        Self {
            data_source,
            data_location,
            connection_string,
            cap_data_rows,
            max_data_rows,
            columns,
            sequence_data,
            sequence_length,
            label_index,
            learning_task,
            training_data,
            training_labels,
            validation_data,
            validation_labels,
            validation_split,
            scaling_factor,
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

    fn load_from_redis(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", error_log_location);
        let connection_str = self.connection_string
            .as_deref()
            .ok_or("Missing connection string for Redis")?;

        let client = redis::Client::open(connection_str)?;
        let mut con = client.get_connection()?;

        let keys: RedisResult<Vec<String>> = con.keys("*");
        if keys.is_err() {
            return Err("No keys found in Redis.".into());
        }

        let mut raw_data_values = Vec::new();
        let mut labels = Vec::new();
        let mut categorical_values = Vec::new();

        let mut row_count = 0;
        let mut skipped_rows = 0;
        let mut num_features = None;

        for key in keys.unwrap() {
            if self.cap_data_rows && row_count >= self.max_data_rows {
                break;
            }

            let value: String = con.get(&key)?;
            let record: Vec<String> = serde_json::from_str(&value)?;

            let (valid, features, label, category) = self.process_record(&record);

            if valid {
                if num_features.is_none() {
                    num_features = Some(features.len()); // Set feature count based on first valid row
                } else if features.len() != num_features.unwrap() {
                    return Err(format!(
                        "Inconsistent feature count: expected {}, found {}",
                        num_features.unwrap(),
                        features.len()
                    )
                    .into());
                }

                raw_data_values.extend_from_slice(&features);
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

        if row_count == 0 {
            return Err("No valid data found in Redis.".into());
        }

        // Convert `raw_data_values` into a `Matrix`
        let raw_data = Matrix::new(row_count, num_features.unwrap(), raw_data_values);

        self.process_loaded_data(raw_data, labels, categorical_values)
    }

    // Load data from PostgreSQL
    fn load_from_postgres(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", error_log_location);
        // Ensure columns configuration is available
        let columns = self.columns.as_ref()
            .ok_or("Error: Columns are required for structured datasets in PostgreSQL.")?;

        let connection_str = self.connection_string
            .as_deref()
            .ok_or("Missing connection string for PostgreSQL")?;

        let mut client = Client::connect(connection_str, NoTls)?;

        let feature_columns = columns.features.join(", ");
        let target_column = &columns.target;
        let categorical_column = &columns.categorical_column;

        let query = format!(
            "SELECT {}, {}, {} FROM my_table",
            feature_columns, target_column, categorical_column
        );

        let rows = client.query(query.as_str(), &[])?;

        let mut raw_data_values = Vec::new();
        let mut labels = Vec::new();
        let mut categorical_values = Vec::new();

        let mut row_count = 0;
        let mut skipped_rows = 0;
        let mut num_features = None;

        for row in rows {
            if self.cap_data_rows && row_count >= self.max_data_rows {
                break;
            }

            // Collect values into a Vec<String>
            let record: Vec<String> = (0..columns.features.len() + 2)
                .map(|i| row.get::<_, String>(i))
                .collect();

            let (valid, features, label, category) = self.process_record(&record);

            if valid {
                if num_features.is_none() {
                    num_features = Some(features.len()); // Set expected feature count
                } else if features.len() != num_features.unwrap() {
                    return Err(format!(
                        "Inconsistent feature count: expected {}, found {}",
                        num_features.unwrap(),
                        features.len()
                    ).into());
                }

                raw_data_values.extend_from_slice(&features);
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

        if row_count == 0 {
            return Err("No valid data found in PostgreSQL.".into());
        }

        // Convert `raw_data_values` into a `Matrix`
        let raw_data = Matrix::new(row_count, num_features.unwrap(), raw_data_values);

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

        // Check if columns exist before using them
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

        // Extract target value safely
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

        // Extract categorical value safely
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
        raw_data: Matrix,
        labels: Vec<f64>,
        categorical_values: Vec<String>
    ) -> Result<(), Box<dyn std::error::Error>> {

        // is raw_data empty
        if raw_data.is_empty() {
            return Err("No valid data to process".into());
        }

        // Encode categorical values (if applicable)
        let categorical_indices = if !categorical_values.is_empty() {
            self.encode_categorical_values(&categorical_values)
        } else {
            Vec::new()
        };


        // Create sequences if enabled
        let (data, sequence_labels) = if self.sequence_data {
            if categorical_indices.is_empty() {
                self.create_sequences_without_categorical(&raw_data, &labels)
            } else {
                self.create_sequences_with_categorical(&raw_data, &labels, &categorical_indices)
            }
        } else {
            //other wise proceed with the raw data
            (raw_data.clone(), labels.clone())
        };

        // Step 4: Finalize data (split into training/validation)
        self.finalize_data(data, sequence_labels)

    }

    fn encode_categorical_values(&self, categorical_values: &[String]) -> Vec<usize> {
        let categorical_map: std::collections::HashMap<String, usize> = categorical_values
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, value)| (value, idx))
            .collect();

        categorical_values
            .iter()
            .map(|value| *categorical_map.get(value).unwrap())
            .collect()
    }

    fn create_sequences_without_categorical(
        &self,
        raw_data: &Matrix,
        labels: &[f64]
    ) -> (Matrix, Vec<f64>) {
        let num_sequences = raw_data.rows - self.sequence_length;
        let sequence_length = self.sequence_length * raw_data.cols; // Total features per sequence

        let mut sequence_data = Vec::with_capacity(num_sequences * sequence_length);
        let mut sequence_labels = Vec::with_capacity(num_sequences);

        for i in 0..num_sequences {
            let sequence_matrix = raw_data.extract_rows(i, i + self.sequence_length);
            sequence_data.extend_from_slice(&sequence_matrix.data); // Flatten
            sequence_labels.push(labels[i + self.sequence_length - 1]);
        }

        (Matrix::new(num_sequences, sequence_length, sequence_data), sequence_labels)
    }

    fn create_sequences_with_categorical(
        &self,
        raw_data: &Matrix,
        labels: &[f64],
        categorical_indices: &[usize]
    ) -> (Matrix, Vec<f64>) {
        let num_sequences = raw_data.rows - self.sequence_length;
        let feature_dim = raw_data.cols + 1; // Extra column for categorical feature
        let sequence_length = self.sequence_length * feature_dim; // Total features per sequence

        let mut sequence_data = Vec::with_capacity(num_sequences * sequence_length);
        let mut sequence_labels = Vec::with_capacity(num_sequences);

        for i in 0..num_sequences {
            let sequence_matrix = raw_data.extract_rows(i, i + self.sequence_length);
            
            // Flatten the extracted matrix into the sequence vector
            for row_idx in 0..sequence_matrix.rows {
                sequence_data.extend_from_slice(&sequence_matrix.row_slice(row_idx).unwrap());
                sequence_data.push(categorical_indices[i + row_idx] as f64); // Append categorical index
            }

            sequence_labels.push(labels[i + self.sequence_length - 1]);
        }

        (Matrix::new(num_sequences, sequence_length, sequence_data), sequence_labels)
    }

    fn finalize_data(
        &mut self,
        mut raw_data: Matrix,
        sequence_labels: Vec<f64>
    ) -> Result<(), Box<dyn std::error::Error>> {

        let mut raw_labels = if self.learning_task == config::LearningTask::Classification {
            Labels::Classification(sequence_labels.iter().map(|&x| x as usize).collect())
        } else {
            Labels::Regression(sequence_labels)
        };

        println!("Shuffling data... {} rows x {} cols", raw_data.rows, raw_data.cols);
        DataLoader::shuffle_data(&mut raw_data, &mut raw_labels);

        println!("Splitting data...");
        if self.validation_split > 0.0 {
            let (training_labels, validation_labels) = self.split_data(&raw_data, &raw_labels);
            self.training_labels = training_labels;
            self.validation_labels = validation_labels;
        } else {
            self.split_index = raw_data.rows;
            self.training_data = raw_data;
            self.training_labels = raw_labels;
        }

        Ok(())
    }

    pub fn load_from_file(&mut self, error_log_location: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", error_log_location);

        // Extract columns before calling `self.load_for_columns`
        let columns = self.columns.clone();

        if let Some(columns) = columns {
            // Pass `columns`, avoiding borrowing `self` mutably while it's already borrowed immutably
            self.load_for_columns(&columns)
        } else {
            // Call `load_for_fixed()` safely
            self.load_for_fixed()
        }
    }

    // Function to load structured datasets (CSV with defined columns)
    fn load_for_columns(&mut self, columns: &config::ColumnsConfig) -> Result<(), Box<dyn std::error::Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&self.data_location)?;

        let headers = reader.headers()?.clone();

        let feature_indices: Vec<usize> = headers.iter()
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

        let mut raw_data_values = Vec::new();
        let mut labels = Vec::new();
        let mut categorical_values = Vec::new();

        let mut row_count = 0;
        let mut skipped_rows = 0;
        let mut num_features = None;

        for record in reader.records() {
            let record = record?;

            if self.cap_data_rows && row_count > self.max_data_rows {
                break;
            }

            let mut valid = true;
            let mut errors = Vec::new();

            let features: Vec<f64> = feature_indices.iter()
                .map(|&idx| record[idx].parse::<f64>()
                    .map(|val| val / self.scaling_factor) // Apply scaling factor here
                    .map_err(|_| format!("Invalid feature value in column {}", idx))
                )
                .filter_map(|res| match res {
                    Ok(val) => Some(val),
                    Err(e) => {
                        valid = false;
                        errors.push(e);
                        None
                    }
                })
                .collect();

            if num_features.is_none() {
                num_features = Some(features.len()); // Set feature count based on first valid row
            } else if features.len() != num_features.unwrap() {
                return Err(format!(
                    "Inconsistent feature count: expected {}, found {}",
                    num_features.unwrap(),
                    features.len()
                ).into());
            }

            let label = record[target_index].parse::<f64>()
                .map(|val| val / self.scaling_factor) // Apply scaling factor here
                .unwrap_or_else(|_| {
                    valid = false;
                    errors.push(format!("Invalid target value in column {}", target_index));
                    0.0
                });

            let categorical_value = record[categorical_index].to_string();

            if !valid {
                skipped_rows += 1;
                println!("Skipping row {} due to errors: {:?}", row_count, errors);
                continue;
            } else {
                raw_data_values.extend_from_slice(&features);
                labels.push(label);
                categorical_values.push(categorical_value);
                row_count += 1;
            }

        } // process records in data file

        println!(
            "Loading {} rows. Skipped {} invalid rows.",
            row_count, skipped_rows
        );

        if row_count == 0 {
            return Err("No valid data found in CSV file.".into());
        }

        // Convert collected values into a `Matrix`
        let raw_data = Matrix::new(row_count, num_features.unwrap(), raw_data_values);

        self.process_loaded_data(raw_data, labels, categorical_values)

    }

    fn load_for_fixed(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false) // Assume no headers in fixed datasets
            .from_path(&self.data_location)?;

        let mut raw_data_values = Vec::new();
        let mut labels = Vec::new();

        let mut row_count = 0;
        let mut skipped_rows = 0;
        let mut num_features = None;

        for record in reader.records() {
            let record = record?;
            row_count += 1;

            if self.cap_data_rows && row_count > self.max_data_rows {
                break;
            }

            let mut valid = true;
            let mut errors = Vec::new();

            // Determine the label index from config
            let label_index = self.label_index;

            let label = if self.learning_task == config::LearningTask::Unsupervised {
                0.0 // If unsupervised, labels are irrelevant
            } else {
                record.get(label_index)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or_else(|| {
                        valid = false;
                        errors.push(format!("Invalid label value at index {}", label_index));
                        0.0
                    })
            };

            // Extract features, excluding the label column
            let features: Vec<f64> = record.iter()
                .enumerate()
                .filter(|(i, _)| *i != label_index) // Skip the label column
                .map(|(_, value)| value.parse::<f64>().unwrap_or_else(|_| {
                    valid = false;
                    errors.push("Invalid feature value".to_string());
                    0.0
                }) / self.scaling_factor) // Normalize grayscale
                .collect();

            if num_features.is_none() {
                num_features = Some(features.len()); // Set feature count from first valid row
            } else if features.len() != num_features.unwrap() {
                return Err(format!(
                    "Inconsistent feature count: expected {}, found {}",
                    num_features.unwrap(),
                    features.len()
                ).into());
            }

            if !valid {
                skipped_rows += 1;
                println!("Skipping row {} due to errors: {:?}", row_count, errors);
                continue;
            }

            raw_data_values.extend_from_slice(&features);
            labels.push(label);
        }

        println!(
            "Read {} rows. Skipped {} invalid rows.",
            row_count, skipped_rows
        );

        if row_count == 0 || skipped_rows == row_count {
            return Err("No valid data found in fixed dataset.".into());
        }

        // Convert collected values into a `Matrix`
        let raw_data = Matrix::new(row_count, num_features.unwrap(), raw_data_values);

        self.process_loaded_data(raw_data, labels, vec![])
    
    }

    pub fn split_data(&mut self, data: &Matrix, labels: &Labels) -> (Labels, Labels) {
        let split_index = ((1.0 - self.validation_split) * data.rows as f64) as usize;

        assert!(
            split_index > 0 && split_index < data.rows,
            "Invalid split_index: {}. Ensure validation_split is correctly set.",
            split_index
        );

        let data_cols = data.cols;

        self.training_data = Matrix::new(
            split_index,
            data_cols,
            data.data[..(split_index * data_cols)].to_vec(),
        );

        self.validation_data = Matrix::new(
            data.rows - split_index,
            data_cols,
            data.data[(split_index * data_cols)..].to_vec(),
        );

        let (train_labels, val_labels) = match labels {
            Labels::Classification(vec) => {
                let (train, val) = vec.split_at(split_index);
                (Labels::Classification(train.to_vec()), Labels::Classification(val.to_vec()))
            }
            Labels::Regression(vec) => {
                let (train, val) = vec.split_at(split_index);
                (Labels::Regression(train.to_vec()), Labels::Regression(val.to_vec()))
            }
        };

        println!(
            "Data split into training ({}) and validation ({}) sets.",
            self.training_data.rows, self.validation_data.rows
        );

        (train_labels, val_labels)
    }

    pub fn shuffle_data(data: &mut Matrix, labels: &mut Labels) {
        assert_eq!(
            data.rows,
            labels.len(),
            "Mismatch: data rows != labels length"
        );

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..data.rows).collect();
        indices.shuffle(&mut rng);

        // Shuffle the matrix data
        for i in 0..data.rows {
            let swap_idx = indices[i];

            for col in 0..data.cols {
                data.data.swap(i * data.cols + col, swap_idx * data.cols + col);
            }
        }

        // Shuffle labels using the indices
        labels.shuffle_with_indices(&indices);
    }

}


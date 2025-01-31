// matrix.rs
// Richard Dalley

//! # Matrix Module
//!
//! This module provides the implementation of the `Matrix` struct and related traits, including:
//! - `Dot`: For performing dot products
//! - `Outer`: For performing outer products
//!
//! It also defines an enum for specifying row-wise or column-wise operations
//! - pub enum VectorType { Row, Column,}
//! 
//! ## Usage:
//! To use the `Matrix` struct along with åits associated traits and operations, add the following to your code:
//! ```rust
//! use crate::matrix::{Matrix, Dot, Outer, VectorType};
//! ```
//! This will bring the `Matrix` struct as well as the `Dot` and `Outer` traits into scope, allowing you to perform matrix operations like dot products and outer products.

// traits
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Div;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use rand_distr::Normal;
use rand_distr::Distribution;

pub trait Dot<Rhs = Self> {
    type Output;

    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

// define a trait Outer for outer products
pub trait Outer<Rhs = Self> {
    type Output;

    fn outer(&self, rhs: &Rhs) -> Self::Output;
}

//Matrix
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, // Flat vector for matrix elements
}

//Implementation of a matrix specialized for a neural network
impl Matrix {
    // Constructor for a new matrix
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Data size mismatch");
        Self { rows, cols, data }
    }

    pub fn rows_iter(&self) -> impl Iterator<Item = &[f64]> {
        (0..self.rows).map(move |row| {
            let start = row * self.cols;
            let end = start + self.cols;
            &self.data[start..end]
        })
    }


    // Zero-initialized matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // Random-initialized matrix
    pub fn random(rows: usize, cols: usize) -> Self {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let normal = Normal::new(0.0, 1.0).unwrap(); // Mean 0.0, Std Dev 1.0
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Self { rows, cols, data }
    }

    // Initialize_weights - neural network specific function for setting weights for the layers
    pub fn initialize_weights(&mut self, nodes_in_previous_layer: usize) {
        let std_dev = (nodes_in_previous_layer as f64).powf(-0.5); // Calculate standard deviation
        let normal = Normal::new(0.0, std_dev).unwrap(); // Normal distribution with mean 0 and std_dev
        let mut rng = rand::thread_rng();

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i * self.cols + j] = normal.sample(&mut rng);
            }
        }
    }

    // Immutable access to matrix elements
    pub fn at(&self, row: usize, col: usize) -> &f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        &self.data[row * self.cols + col]
    }

    // Return the index of the maximum value in the data
    pub fn argmax(&self) -> usize {
        // Assume it's a vector; precondition checked elsewhere
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or_else(|| panic!("Matrix is empty, cannot compute argmax."))
    }

    // argmax_row
    pub fn argmax_row(slice: &[f64]) -> usize {
        slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .expect("Slice is empty")
    }
    
    pub fn softmax_row(&self, input: &[f64]) -> Vec<f64> {
        let max_input = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = input.iter().map(|&x| (x - max_input).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f64>();
        exp_values.iter().map(|&x| x / sum_exp).collect()
    }

    pub fn softmax(&self) -> Matrix {
        let mut result_data = Vec::new();
        
        for row in 0..self.rows {
            let start = row * self.cols;
            let end = start + self.cols;
            let row_slice = &self.data[start..end]; // Get a row slice
            let processed_row = self.softmax_row(row_slice); // Apply the softmax function on the row
            result_data.extend_from_slice(&processed_row);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }


    // Mutable access to matrix elements
    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        &mut self.data[row * self.cols + col]
    }

    // Transpose - flip rows and cols
    pub fn transpose(&self) -> Self {

        let mut transposed = Matrix::zeros(self.cols, self.rows); // Swap rows and cols

        for i in 0..self.rows {
            for j in 0..self.cols {
                // Transpose logic: element at (i, j) becomes (j, i)
                transposed.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        transposed
    }

    // Copy a row or col from the matrix
    // TODO: this is poorly named
    pub fn extract(&self) -> Result<Vec<f64>, String> {
        if self.rows == 1 {
            // Row vector: return all elements
            Ok(self.data.clone())
        } else if self.cols == 1 {
            // Column vector: return elements in column order
            Ok(self.data.clone())
        } else {
            // Error: matrix is not a vector
            Err("Matrix is not a vector (1 row or 1 column).".to_string())
        }
    }

    // Returns a slice for the row specified by row_index
    pub fn row_slice(&self, row_index: usize) -> Result<&[f64], String> {
        if row_index >= self.rows {
            return Err("Row index out of bounds.".to_string());
        }

        let start = row_index * self.cols;
        let end = start + self.cols;

        Ok(&self.data[start..end]) // Return a slice for the row
    }


    /// Extracts a subset of rows from the matrix and returns a new Matrix.
    ///
    /// # Arguments
    /// * `start` - The starting row index (inclusive).
    /// * `end` - The ending row index (exclusive).
    ///
    /// # Returns
    /// A new `Matrix` containing only the specified rows.
    ///
    /// # Panics
    /// Panics if `start` or `end` is out of bounds or if `start >= end`.
    pub fn extract_rows(&self, start: usize, end: usize) -> Matrix {
        if start >= end || end > self.rows {
            panic!(
                "Row indices out of bounds: start={}, end={}, max={}",
                start, end, self.rows
            );
        }

        let row_count = end - start;
        let mut extracted_data = Vec::with_capacity(row_count * self.cols);

        extracted_data.extend_from_slice(&self.data[start * self.cols..end * self.cols]);

        Matrix::new(row_count, self.cols, extracted_data)
    }


    /// Creates a row matrix (1 row, `vec.len()` columns)
    pub fn from_row(vec: Vec<f64>) -> Self {
        Self {
            rows: 1,
            cols: vec.len(),
            data: vec,
        }
    }

    pub fn repeat_columns(&self, target_cols: usize) -> Matrix {
        assert_eq!(self.cols, 1, "repeat_columns only supports single-column matrices");

        let mut repeated_data = Vec::with_capacity(self.rows * target_cols);

        for i in 0..self.rows {
            let value = self.data[i]; // Get the single value for this row
            repeated_data.extend(std::iter::repeat(value).take(target_cols)); // Repeat it target_cols times
        }

        Matrix::new(self.rows, target_cols, repeated_data)
    }


    /// Creates a one-hot encoded matrix from a set of labels.
    ///
    /// # Arguments
    /// - `labels`: A slice of label indices (e.g., `[0, 2, 1]`).
    /// - `num_classes`: The total number of classes.
    ///
    /// # Returns
    /// A matrix with dimensions `(labels.len(), num_classes)` where each row is
    /// a one-hot vector representing the corresponding label.
    pub fn from_labels(labels: &[usize], num_classes: usize) -> Self {
        let rows = labels.len();
        let cols = num_classes;
        let mut data = vec![0.0; rows * cols];

        for (i, &label) in labels.iter().enumerate() {
            let clamped_label = if label >= num_classes {
            println!(
                "Warning: Label index out of bounds. i={}, label={}, num_classes={}, rows={}, cols={}. Clamping label to {}.",
                i, label, num_classes, rows, cols, num_classes - 1
            );
            num_classes - 1 // Clamp to the maximum valid label
        } else {
            label
        };
            // assert!(label < num_classes, "Label index out of bounds");
            data[i * cols + clamped_label] = 1.0;
        }

        Matrix {
            rows,
            cols,
            data,
        }
    }

    /// Creates a column matrix (`vec.len()` rows, 1 column)
    pub fn from_col(vec: Vec<f64>) -> Self {
        Self {
            rows: vec.len(),
            cols: 1,
            data: vec,
        }
    }


    pub fn column_to_indices(&self, column_index: usize) -> Vec<usize> {
        assert!(
            column_index < self.cols,
            "Column index out of bounds. The matrix has {} columns.",
            self.cols
        );

        // Preallocate the vector for efficiency
        let mut indices = Vec::with_capacity(self.rows);

        // Use an iterator to calculate indices and collect into the result
        indices.extend(
            (0..self.rows).map(|row| unsafe {
                // Directly access without bounds checks
                *self.data.get_unchecked(row * self.cols + column_index) as usize
            }),
        );

        indices
    }

    pub fn slice(&self, start: usize, end: usize) -> Matrix {
        let end = end.min(self.rows);
        Matrix::new(end - start, self.cols, self.data[start * self.cols..end * self.cols].to_vec())
    }

    pub fn one_hot(index: usize, num_classes: usize) -> Matrix {
        let mut data = vec![0.0; num_classes];
        data[index] = 1.0;
        Matrix::new(1, num_classes, data)
    }


    // The attention mechanism involves scaling the dot product of queries (Q) and keys (K) 
    // by the square root of the dimensionality of the keys √dk) to stabilize gradients.
    pub fn scale(&self, scalar: f64) -> Matrix {
        let scaled_data: Vec<f64> = self.data.iter().map(|&x| x / scalar).collect();
        Matrix::new(self.rows, self.cols, scaled_data)
    }

    // Masked self-attention prevents a token from attending to future tokens during training. 
    // This is done by applying a mask that sets certain positions in the attention scores to -∞,
    //  before applying the softmax function.
    pub fn mask(&self, mask: &Matrix, masked_value: f64) -> Matrix {
        assert_eq!(self.rows, mask.rows, "Mask and matrix rows must match");
        assert_eq!(self.cols, mask.cols, "Mask and matrix cols must match");

        let masked_data: Vec<f64> = self
            .data
            .iter()
            .zip(mask.data.iter())
            .map(|(&x, &m)| if m == 1.0 { x } else { masked_value })
            .collect();

        Matrix::new(self.rows, self.cols, masked_data)
    }
    
    pub fn upper_triangular_mask(size: usize) -> Matrix {
        let mut mask_data = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..size {
                if j > i {
                    mask_data[i * size + j] = f64::NEG_INFINITY;
                } else {
                    mask_data[i * size + j] = 1.0;
                }
            }
        }
        Matrix::new(size, size, mask_data)
    }

    // Attention scores require softmax to be applied row-wise.
    pub fn softmax_rows(&self) -> Matrix {
        let epsilon = 1e-9; // Small value for numerical stability
        let mut data = Vec::with_capacity(self.data.len());

        for i in 0..self.rows {
            let row_start = i * self.cols;
            let row_end = row_start + self.cols;

            let row = &self.data[row_start..row_end];
            let max_row = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let exp_row: Vec<f64> = row.iter().map(|&x| (x - max_row).exp()).collect();
            let sum_exp = exp_row.iter().sum::<f64>();

            if sum_exp.abs() < epsilon {
                // Handle zero or near-zero sum by normalizing to a uniform distribution
                data.extend(row.iter().map(|_| 1.0 / self.cols as f64));
                eprintln!("Warning: Softmax encountered a zero or near-zero sum.");
            } else {
                // Perform regular softmax normalization
                data.extend(exp_row.iter().map(|&x| x / sum_exp));
            }
        }

        Matrix::new(self.rows, self.cols, data)
    }

    //  Generate random Q, K, and V matrices for attention layers.
    pub fn random_with_shape(rows: usize, cols: usize) -> Self {
        Self::random(rows, cols)
    }

    pub fn broadcast(&self, rows: usize) -> Matrix {
        assert_eq!(self.rows, 1, "Broadcast only supports single-row matrices");
        let mut data = Vec::new();
        for _ in 0..rows {
            data.extend_from_slice(&self.data);
        }
        Matrix::new(rows, self.cols, data)
    }


    // Add a row vector to every row of the Matrix
    pub fn add_broadcast(&self, vec: &Matrix) -> Matrix {
        assert_eq!(vec.rows, 1, "Vector must have one row for broadcasting.");
        assert_eq!(self.cols, vec.cols, "Vector and matrix columns must match.");

        let mut result = self.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i * self.cols + j] += vec.data[j];
            }
        }
        result
    }

    // Clip values in a matrix to a specified range (to avoid issues with exploding or vanishing gradients)
    pub fn clip(&self, min: f64, max: f64) -> Matrix {
        let clipped_data: Vec<f64> = self.data.iter().map(|&x| x.min(max).max(min)).collect();
        Matrix::new(self.rows, self.cols, clipped_data)
    }

    pub fn add_head(&mut self, head: &Matrix, head_index: usize, head_dim: usize) {
        // Ensure the dimensions match
        assert_eq!(head.cols, head_dim, "Head dimension mismatch");
        assert_eq!(head.rows, self.rows, "Row count mismatch");
        assert!(head_dim * (head_index + 1) <= self.cols, "Head dimension out of bounds");

        // Determine where to place the data
        let start_col = head_index * head_dim;

        // Add the head matrix into the target matrix
        for row in 0..self.rows {
            for col in 0..head.cols {
                self.data[row * self.cols + start_col + col] += head.data[row * head.cols + col];
            }
        }
    }

    pub fn extract_head(&self, head: usize, head_dim: usize) -> Matrix {
        assert!(head_dim * (head + 1) <= self.cols, "Head dimension out of bounds");
        let mut result = Matrix::zeros(self.rows, head_dim);

        for i in 0..self.rows {
            let start = head * head_dim;
            let end = start + head_dim;
            result.data[i * head_dim..(i + 1) * head_dim]
                .copy_from_slice(&self.data[i * self.cols + start..i * self.cols + end]);
        }

        result
    }

    pub fn concat_heads(heads: &[Matrix]) -> Matrix {
        let rows = heads[0].rows;
        let cols: usize = heads.iter().map(|h| h.cols).sum();
        let mut concatenated = Matrix::zeros(rows, cols);

        for (head_idx, head) in heads.iter().enumerate() {
            for i in 0..rows {
                let start_col = head_idx * head.cols;
                let end_col = start_col + head.cols;
                concatenated.data[i * cols + start_col..i * cols + end_col]
                    .copy_from_slice(&head.data[i * head.cols..(i + 1) * head.cols]);
            }
        }

        concatenated
    }

    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    pub fn mean_axis(&self, axis: usize) -> Matrix {
        match axis {
            0 => {
                // Mean along rows
                let mut result = vec![0.0; self.cols];
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        result[col] += self.data[row * self.cols + col];
                    }
                }
                for col in 0..self.cols {
                    result[col] /= self.rows as f64;
                }
                Matrix::new(1, self.cols, result)
            }
            _ => panic!("Unsupported axis for mean_axis"),
        }
    }

    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.data.len() as f64
    }


    // Apply the funcion to the data
    pub fn apply<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| func(x)).collect();
        Matrix::new(self.rows, self.cols, data)
    }

    pub fn standard_dev(&self, axis: usize, means: Option<&Matrix>) -> Matrix {
        match axis {
            0 => {
                // Standard deviation along columns
                let mut result = vec![0.0; self.cols];
                let feature_means = match means {
                    Some(m) => &m.data,  // Use provided means
                    None => &self.mean_axis(0).data, // Compute if not provided
                };

                for row in 0..self.rows {
                    for col in 0..self.cols {
                        let diff = self.data[row * self.cols + col] - feature_means[col];
                        result[col] += diff.powi(2);
                    }
                }

                for col in 0..self.cols {
                    result[col] = (result[col] / self.rows as f64).sqrt();
                }

                Matrix::new(1, self.cols, result)
            }
            _ => panic!("Unsupported axis for standard deviation"),
        }
    }

    /// Normalizes the matrix using given means and standard deviations.
    /// If means and standard deviations are not provided, they are computed internally.
    pub fn normalize(&mut self, means: Option<&Matrix>, stds: Option<&Matrix>) {
        let feature_means = match means {
            Some(m) => m.clone(),          // Use provided means
            None => self.mean_axis(0),     // Compute means if not provided
        };

        let feature_stds = match stds {
            Some(s) => s.clone(),          // Use provided standard deviations
            None => self.standard_dev(0, Some(&feature_means)), // Compute std if not provided
        };

        // Apply normalization: (X - mean) / std
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = row * self.cols + col;
                self.data[index] = (self.data[index] - feature_means.data[col]) / feature_stds.data[col];
            }
        }
    }


    // Print the matrices in a readable format
    pub fn pretty_print(&self) {
        for i in 0..self.rows {
            let row: Vec<_> = self.data[i * self.cols..(i + 1) * self.cols]
                .iter()
                .map(|x| format!("{:8.4}", x))
                .collect();
            println!("[{}]", row.join(", "));
        }
    }

}


// Implement element-wise multiplication for Matrix
// Owned × Owned
impl Mul for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Reference × Reference
impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Owned × Reference
impl<'a> Mul<&'a Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'a Matrix) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Reference × Owned
impl<'a> Mul<Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Implement Scalar multiplication for Matrix
impl Mul<f64> for Matrix {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Matrix::new(self.rows, self.cols, data)
    }
}

impl<'a> Mul<f64> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Matrix::new(self.rows, self.cols, data)
    }
}

// Implement Scalar multiplication assgn (*=) for Matrix
impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, scalar: f64) {
        self.data.iter_mut().for_each(|x| *x *= scalar); // In-place scaling
    }
}

impl Dot for Matrix {
    type Output = Matrix;

    fn dot(&self, other: &Matrix) -> Self::Output {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch for dot product");

        let block_size = 64; // Cache-friendly block size (adjust as needed)
        let mut result = Matrix::zeros(self.rows, other.cols);

        for i_block in (0..self.rows).step_by(block_size) {
            for j_block in (0..other.cols).step_by(block_size) {
                for k_block in (0..self.cols).step_by(block_size) {
                    for i in i_block..(i_block + block_size).min(self.rows) {
                        for j in j_block..(j_block + block_size).min(other.cols) {
                            let mut sum = 0.0;
                            for k in k_block..(k_block + block_size).min(self.cols) {
                                sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                            }
                            result.data[i * other.cols + j] += sum;
                        }
                    }
                }
            }
        }

        result
    }

}



// Implement Outer product for matrix
impl Outer for Matrix {
    type Output = Matrix;

    fn outer(&self, other: &Matrix) -> Self::Output {
        // Ensure `self` is a column vector
        if self.cols != 1 {
            panic!("First matrix must be a column vector for outer product.");
        }

        // Ensure `other` is a row vector
        if other.rows != 1 {
            panic!("Second matrix must be a row vector for outer product.");
        }

        // Create result matrix with dimensions (self.rows x other.cols)
        let mut result = Matrix::zeros(self.rows, other.cols);

        // Perform outer product
        for i in 0..self.rows {
            for j in 0..other.cols {
                result.data[i * other.cols + j] = self.data[i] * other.data[j];
            }
        }

        result
    }
}

// Implement Add for Matrix
impl Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b) // Element-wise addition
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Implement Add for &Matrix
impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b) // Element-wise addition
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Implement append (AddAssign) for Matrix
impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        self.data.iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(a, b)| *a += b); // Element-wise addition, in-place
    }
}
// Sub for owned matrices (Matrix - Matrix)
impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Sub for borrowing a matrix (&Matrix - &Matrix)
impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &'b Matrix) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Sub for owned Matrix with a reference Matrix (Matrix - &Matrix)
impl<'a> Sub<&'a Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &'a Matrix) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Sub for borrowing Matrix with an owned Matrix (&Matrix - Matrix)
impl<'a> Sub<Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix::new(self.rows, self.cols, data)
    }
}

// Implement remove (SubAssign -=) for Matrix
impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "Row dimensions must match");
        assert_eq!(self.cols, rhs.cols, "Column dimensions must match");

        self.data.iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(a, b)| *a -= b); // In-place element-wise subtraction
    }
}

impl Div<f64> for Matrix {
    type Output = Matrix;

    fn div(self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|&x| x / scalar).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<'a> Div<f64> for &'a Matrix {
    type Output = Matrix;

    fn div(self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|&x| x / scalar).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
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


// enums
pub enum VectorType {
    Row,
    Column,
}


//Matrix
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, // Flat vector for matrix elements
}

//IMplementation of a matrix specialized for a neural network
impl Matrix {
    // Constructor for a new matrix
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Data size mismatch");
        Self { rows, cols, data }
    }

    pub fn new_from_vector(vec: Vec<f64>, vector_type: VectorType) -> Self {
        match vector_type {
            VectorType::Column => Self {
                rows: vec.len(),
                cols: 1,
                data: vec,
            },
            VectorType::Row => Self {
                rows: 1,
                cols: vec.len(),
                data: vec,
            },
        }
    }

    // Zero-initialized matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
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

    // initialize_weights - neural network specific function for setting weights for the layers
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

    // Mutable access to matrix elements
    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        &mut self.data[row * self.cols + col]
    }

    // transpose - flip rows and cols
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

    // extract - copy a row or col from the matrix
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

    pub fn row_slice(&self, row_index: usize) -> Result<&[f64], String> {
        if row_index >= self.rows {
            return Err("Row index out of bounds.".to_string());
        }

        let start = row_index * self.cols;
        let end = start + self.cols;

        Ok(&self.data[start..end]) // Return a slice for the row
    }

    pub fn from_vector(vec: Vec<f64>, orientation: VectorType) -> Self {
        match orientation {
            VectorType::Column => {
                let rows = vec.len();
                let cols = 1;
                Matrix {
                    rows,
                    cols,
                    data: vec,
                }
            }
            VectorType::Row => {
                let rows = 1;
                let cols = vec.len();
                Matrix {
                    rows,
                    cols,
                    data: vec,
                }
            }
        }
    }

    pub fn argmax(&self) -> usize {
        // Ensure the matrix is a column vector (1D vector)
        if self.rows != 1 && self.cols != 1 {
            panic!("argmax is only valid for vectors (1 row or 1 column).");
        }

        // Find the index of the maximum value in the data
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .expect("Matrix is empty, cannot compute argmax.")
    }

    //apply the funcion to the data
    pub fn apply<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| func(x)).collect();
        Matrix::new(self.rows, self.cols, data)
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

// Implement Scalar multiplication assgn (*=) for Matrix
impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, scalar: f64) {
        self.data.iter_mut().for_each(|x| *x *= scalar); // In-place scaling
    }
}

// Implement dot product for Matrix
impl Dot for Matrix {
    type Output = Matrix;

    fn dot(&self, other: &Matrix) -> Self::Output {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch for dot product");

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
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

// Implement Subtract for Matrix
impl<T> Sub<T> for Matrix
where
    T: std::ops::Deref<Target = Matrix>,
{
    type Output = Matrix;

    fn sub(self, rhs: T) -> Self::Output {
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

// Sub - returns the difference between 2 matrices
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

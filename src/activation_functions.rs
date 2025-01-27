
use crate::config;

// ActivationFn
pub type ActivationFn = fn(f64) -> f64;

pub trait ActivationTrait {
    fn apply(&self, x: f64) -> f64;
}
pub fn get_activation_and_derivative( config: &config::Config) -> (Box<dyn ActivationTrait>, Box<dyn ActivationTrait>) {
        match config.activation_fn_name.to_lowercase().as_str() {
            "sigmoid" => (
                Box::new(Sigmoid),
                Box::new(SigmoidDerivative),
            ),
            "swish" => (
                Box::new(Swish),
                Box::new(SwishDerivative), // Assuming you implement this
            ),
            "relu" => (
                Box::new(ReLU),
                Box::new(ReLUDerivative),
            ),
            "leaky_relu" => (
                Box::new(LeakyReLU { alpha: config.activation_alpha }),
                Box::new(LeakyReLUDerivative { alpha: config.activation_alpha }),
            ),
            "elu" => (
                Box::new(ELU { alpha: config.activation_alpha }),
                Box::new(ELUDerivative { alpha: config.activation_alpha }),
            ),
            "gelu" => (
                Box::new(GELU),
                Box::new(GELUDerivative), // Assuming you implement this
            ),
            "softplus" => (
                Box::new(Softplus),
                Box::new(SoftplusDerivative), // Assuming you implement this
            ),
            "silu" => (
                Box::new(SiLU),
                Box::new(SiLUDerivative), // Assuming you implement this
            ),
            "mish" => (
                Box::new(Mish),
                Box::new(MishDerivative), // Assuming you implement this
            ),
            "hardswish" => (
                Box::new(HardSwish),
                Box::new(HardSwishDerivative), // Assuming you implement this
            ),
            "softsign" => (
                Box::new(Softsign),
                Box::new(SoftsignDerivative), // Assuming you implement this
            ),
            "prelu" => (
                Box::new(PReLU { alpha: config.activation_alpha }),
                Box::new(PReLUDerivative { alpha: config.activation_alpha }), // Assuming you implement this
            ),
            "selu" => (
                Box::new(SELU { alpha: config.activation_alpha, lambda: config.activation_lambda }),
                Box::new(SELUDerivative { alpha: config.activation_alpha, lambda: config.activation_lambda }), // Assuming you implement this
            ),
            _ => panic!("Unknown activation function: {}", config.activation_fn_name.to_lowercase().as_str()),
        }
    }
pub struct Sigmoid;

impl ActivationTrait for Sigmoid {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
} 

pub struct SigmoidDerivative;
impl ActivationTrait for SigmoidDerivative {
    fn apply(&self,x: f64) -> f64 {
        let s = 1.0 / (1.0 + (-x).exp());
        s * (1.0 - s)
    }
}

pub struct Swish;
impl ActivationTrait for Swish{
    fn apply(&self, x: f64) -> f64 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}


pub struct SwishDerivative;

impl ActivationTrait for SwishDerivative {
    fn apply(&self, x: f64) -> f64 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}

pub struct ReLU;
impl ActivationTrait for ReLU{
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

}

pub struct LeakyReLU {
    pub alpha:f64,
}

impl ActivationTrait for LeakyReLU{
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { self.alpha * x }
    }
}

pub struct ELU {
    pub alpha:f64,
}
impl ActivationTrait for ELU{
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { self.alpha * (x.exp() - 1.0) }
    }
}


pub struct LeakyReLUDerivative {
   pub  alpha:f64,
}
impl ActivationTrait for LeakyReLUDerivative{
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { self.alpha }
    }
}

pub struct ELUDerivative {
    pub alpha:f64,
}
impl ActivationTrait for ELUDerivative{
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { self.alpha * x.exp() }
    }
}

pub struct TanH;
impl ActivationTrait for TanH{
    fn apply(&self, x: f64) -> f64 {
        x.tanh()
    }
}

pub struct ReLUDerivative;
impl ActivationTrait for ReLUDerivative{
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct GELU;

impl ActivationTrait for GELU {
    fn apply(&self, x: f64) -> f64 {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let cdf = 0.5 * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh());
        x * cdf
    }
}

pub struct GELUDerivative;

impl ActivationTrait for GELUDerivative {
    fn apply(&self, x: f64) -> f64 {
        // Compute the CDF (Φ(x)) using the error function (erf)
        let cdf = 0.5 * (1.0 + (x / (2.0f64).sqrt()).tanh()); // Approximation of Φ(x)

        // Compute the PDF (ϕ(x))
        let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();

        // GELU derivative: Φ(x) + x * ϕ(x)
        cdf + x * pdf
    }
}

pub struct Softplus;

impl ActivationTrait for Softplus {
    fn apply(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }
}

pub struct SoftplusDerivative;

impl ActivationTrait for SoftplusDerivative {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

pub struct SiLU;

impl ActivationTrait for SiLU {
    fn apply(&self, x: f64) -> f64 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}

pub struct SiLUDerivative;

impl ActivationTrait for SiLUDerivative {
    fn apply(&self, x: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let silu = x * sigmoid;
        silu + sigmoid * (1.0 - silu)
    }
}

pub struct Mish;

impl ActivationTrait for Mish {
    fn apply(&self, x: f64) -> f64 {
        x * (1.0 + x.exp()).ln().tanh()
    }
}

pub struct MishDerivative;

impl ActivationTrait for MishDerivative {
    fn apply(&self, x: f64) -> f64 {
        let softplus = (1.0 + x.exp()).ln();
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let sech = 1.0 / softplus.cosh();
        sigmoid * softplus.tanh() + x * sigmoid * (1.0 - sigmoid) * sech * sech
    }
}


pub struct HardSwish;

impl ActivationTrait for HardSwish {
    fn apply(&self, x: f64) -> f64 {
        x * (x + 3.0).max(0.0).min(6.0) / 6.0
    }
}
pub struct HardSwishDerivative;

impl ActivationTrait for HardSwishDerivative {
    fn apply(&self, x: f64) -> f64 {
        if x > 3.0 {
            1.0
        } else if x >= -3.0 {
            (x + 3.0) / 6.0
        } else {
            0.0
        }
    }
}

pub struct Softsign;

impl ActivationTrait for Softsign {
    fn apply(&self, x: f64) -> f64 {
        x / (1.0 + x.abs())
    }
}

pub struct SoftsignDerivative;

impl ActivationTrait for SoftsignDerivative {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + x.abs()).powi(2)
    }
}


pub struct PReLU {
    pub alpha: f64,
}

impl ActivationTrait for PReLU {
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { self.alpha * x }
    }
}

pub struct PReLUDerivative {
    pub alpha: f64,
}

impl ActivationTrait for PReLUDerivative {
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { self.alpha }
    }
}


pub struct SELU {
    pub alpha: f64,
    pub lambda: f64,
}

impl ActivationTrait for SELU {
    // recommended values for alpha and lambda:
    // α ≈ 1.6732632423543772
    // λ ≈ 1.0507009873554802
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 {
            self.lambda * x
        } else {
            self.lambda * (self.alpha * x.exp() - self.alpha)
        }
    }
}

pub struct SELUDerivative {
    pub alpha: f64,
    pub lambda: f64,
}

impl ActivationTrait for SELUDerivative {
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 {
            self.lambda
        } else {
            self.lambda * self.alpha * x.exp()
        }
    }
}
                
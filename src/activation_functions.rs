
// ActivationFn
pub type ActivationFn = fn(f64) -> f64;

pub trait ActivationTrait {
    fn apply(&self, x: f64) -> f64;
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

pub struct Softplus;

impl ActivationTrait for Softplus {
    fn apply(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }
}

pub struct SiLU;

impl ActivationTrait for SiLU {
    fn apply(&self, x: f64) -> f64 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}

pub struct Mish;

impl ActivationTrait for Mish {
    fn apply(&self, x: f64) -> f64 {
        x * (1.0 + x.exp()).ln().tanh()
    }
}

pub struct HardSwish;

impl ActivationTrait for HardSwish {
    fn apply(&self, x: f64) -> f64 {
        x * (x + 3.0).max(0.0).min(6.0) / 6.0
    }
}

pub struct Softsign;

impl ActivationTrait for Softsign {
    fn apply(&self, x: f64) -> f64 {
        x / (1.0 + x.abs())
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

pub struct SELU {
    pub alpha: f64,
    pub lambda: f64,
}

impl ActivationTrait for SELU {
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 {
            self.lambda * x
        } else {
            self.lambda * (self.alpha * x.exp() - self.alpha)
        }
    }
}
use rand::Rng;
use std::f64::consts::PI;

pub struct QuantumNeuron {
    phase: f64,
    superposition: bool,
}

impl QuantumNeuron {
    pub fn new() -> Self {
        QuantumNeuron {
            phase: 0.0,
            superposition: false,
        }
    }

    pub fn activate(&mut self, input: f64, emotional_state: f64) -> f64 {
        let mut rng = rand::thread_rng();
        
        self.phase += input * PI * 2.0;
        self.phase %= 2.0 * PI;

        if rng.gen::<f64>() < emotional_state {
            self.superposition = !self.superposition;
        }

        if self.superposition {
            (self.phase.sin() + self.phase.cos()) / 2.0
        } else {
            self.phase.sin()
        }
    }

    pub fn calculate_gradient(&self, error: f64) -> f64 {
        if self.superposition {
            error * (self.phase.cos() - self.phase.sin()) / 2.0
        } else {
            error * self.phase.cos()
        }
    }
}
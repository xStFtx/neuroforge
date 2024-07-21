use rand::Rng;
use ndarray::{Array, Array1, Array2};

pub mod adaptive_architecture;
pub mod quantum_neuron;
pub mod emotional_memory;
pub mod temporal_plasticity;
pub mod neuro_symbolic;

use crate::quantum_neuron::QuantumNeuron;
use crate::adaptive_architecture::AdaptiveLayer;
use crate::temporal_plasticity::TemporalNeuron;
use crate::emotional_memory::EmotionalMemory;
use crate::neuro_symbolic::NeuroSymbolicLayer;

pub struct NeuroForge {
    quantum_layers: Vec<QuantumLayer>,
    adaptive_layers: Vec<AdaptiveLayer>,
    temporal_layers: Vec<TemporalLayer>,
    emotional_memory: EmotionalMemory,
    neuro_symbolic_layer: NeuroSymbolicLayer,
    emotional_state: f64,
}

struct QuantumLayer {
    neurons: Vec<QuantumNeuron>,
    weights: Array2<f64>,
}

struct TemporalLayer {
    neurons: Vec<TemporalNeuron>,
}

impl NeuroForge {
    pub fn new(layer_sizes: &[usize], adaptive_layers: &[bool], temporal_layers: &[bool]) -> Self {
        let mut quantum_layers = Vec::new();
        let mut adaptive_layers_vec = Vec::new();
        let mut temporal_layers_vec = Vec::new();

        for (&size, (&is_adaptive, &is_temporal)) in layer_sizes.iter().zip(adaptive_layers.iter().zip(temporal_layers.iter())) {
            if is_adaptive {
                adaptive_layers_vec.push(AdaptiveLayer::new(size, size * 2, size / 2, 0.1));
            } else if is_temporal {
                temporal_layers_vec.push(TemporalLayer::new(size));
            } else {
                quantum_layers.push(QuantumLayer::new(size));
            }
        }

        NeuroForge {
            quantum_layers,
            adaptive_layers: adaptive_layers_vec,
            temporal_layers: temporal_layers_vec,
            emotional_memory: EmotionalMemory::new(100),
            neuro_symbolic_layer: NeuroSymbolicLayer::new(),
            emotional_state: 0.5,
        }
    }

    pub fn forward(&mut self, input: &[f64], time: f64) -> Vec<f64> {
        let mut current_input = input.to_vec();

        for layer in &mut self.quantum_layers {
            current_input = layer.forward(&current_input, self.emotional_state);
        }

        for layer in &mut self.adaptive_layers {
            current_input = layer.forward(&current_input);
        }

        for layer in &mut self.temporal_layers {
            current_input = layer.forward(&current_input, time);
        }

        current_input = self.neuro_symbolic_layer.process(current_input);

        self.emotional_memory.store(current_input.clone(), self.emotional_state);

        current_input
    }

    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = self.forward(input, 0.0);
                total_error += self.backward(target, learning_rate);
                self.update_emotional_state(&output, target);
                self.adapt_architecture();
            }
            println!("Epoch {}: error = {}", epoch, total_error / inputs.len() as f64);
        }
    }

    fn backward(&mut self, target: &[f64], learning_rate: f64) -> f64 {
        let mut current_error = target.to_vec();
        let mut total_error = 0.0;

        current_error = self.neuro_symbolic_layer.backward(&current_error);

        for layer in self.temporal_layers.iter_mut().rev() {
            current_error = layer.backward(&current_error, learning_rate);
        }

        for layer in self.adaptive_layers.iter_mut().rev() {
            current_error = layer.backward(&current_error, learning_rate);
        }

        for layer in self.quantum_layers.iter_mut().rev() {
            current_error = layer.backward(&current_error, learning_rate);
        }

        total_error = current_error.iter().map(|&e| e.powi(2)).sum::<f64>() / current_error.len() as f64;

        total_error
    }

    fn update_emotional_state(&mut self, output: &[f64], target: &[f64]) {
        let error: f64 = output.iter().zip(target.iter()).map(|(o, t)| (o - t).powi(2)).sum::<f64>() / output.len() as f64;
        self.emotional_state = 0.9 * self.emotional_state + 0.1 * error;
    }

    fn adapt_architecture(&mut self) {
        for layer in &mut self.adaptive_layers {
            layer.adapt(self.emotional_state);
        }
    }
}

impl QuantumLayer {
    fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        QuantumLayer {
            neurons: (0..size).map(|_| QuantumNeuron::new()).collect(),
            weights: Array::from_shape_fn((size, size), |_| rng.gen_range(-1.0..1.0)),
        }
    }

    fn forward(&mut self, input: &[f64], emotional_state: f64) -> Vec<f64> {
        let input_array = Array1::from_vec(input.to_vec());
        let weighted_inputs = self.weights.dot(&input_array);
        
        self.neurons
            .iter_mut()
            .zip(weighted_inputs.iter())
            .map(|(neuron, &input)| neuron.activate(input, emotional_state))
            .collect()
    }

    fn backward(&mut self, error: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut next_error = vec![0.0; self.weights.shape()[1]];
        let mut weight_gradients = Array2::zeros(self.weights.dim());

        for (i, (neuron, &neuron_error)) in self.neurons.iter_mut()
            .zip(error.iter()).enumerate() {
            let gradient = neuron.calculate_gradient(neuron_error);
            for j in 0..self.weights.shape()[1] {
                let input = next_error[j];
                weight_gradients[[i, j]] = gradient * input;
                next_error[j] += neuron_error * self.weights[[i, j]];
            }
        }

        self.weights -= &(weight_gradients * learning_rate);

        next_error
    }
}

impl TemporalLayer {
    fn new(size: usize) -> Self {
        TemporalLayer {
            neurons: (0..size).map(|_| TemporalNeuron::new(size)).collect(),
        }
    }

    fn forward(&mut self, input: &[f64], time: f64) -> Vec<f64> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.activate(input, time))
            .collect()
    }

    fn backward(&mut self, error: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut next_error = vec![0.0; self.neurons[0].input_size()];

        for (neuron, &neuron_error) in self.neurons.iter_mut().zip(error.iter()) {
            let neuron_gradients = neuron.calculate_gradients(neuron_error);
            neuron.update_weights(&neuron_gradients, learning_rate);

            for (i, &gradient) in neuron_gradients.iter().enumerate() {
                next_error[i] += gradient;
            }
        }

        next_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuroforge_creation() {
        let network = NeuroForge::new(&[2, 3, 1], &[false, false, false], &[false, false, false]);
        assert_eq!(network.quantum_layers.len(), 3);
        assert!(network.adaptive_layers.is_empty());
        assert!(network.temporal_layers.is_empty());
    }

    #[test]
    fn test_forward_pass() {
        let mut network = NeuroForge::new(&[2, 3, 1], &[false, false, false], &[false, false, false]);
        let input = vec![1.0, 0.0];
        let output = network.forward(&input, 0.0);
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_training() {
        let mut network = NeuroForge::new(&[2, 3, 1], &[false, false, false], &[false, false, false]);
        let inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        network.train(&inputs, &targets, 1000, 0.1);
        // Check if the network has learned XOR function (approximately)
        for (input, expected) in inputs.iter().zip(targets.iter()) {
            let output = network.forward(input, 0.0);
            assert!((output[0] - expected[0]).abs() < 0.1);
        }
    }
}
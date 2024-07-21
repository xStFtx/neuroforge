use rand::Rng;
use std::collections::VecDeque;

pub struct AdaptiveLayer {
    neurons: Vec<AdaptiveNeuron>,
    max_neurons: usize,
    min_neurons: usize,
    adaptation_threshold: f64,
}

struct AdaptiveNeuron {
    weights: Vec<f64>,
    activation_history: VecDeque<f64>,
    importance_score: f64,
}

impl AdaptiveLayer {
    pub fn new(initial_neurons: usize, max_neurons: usize, min_neurons: usize, adaptation_threshold: f64) -> Self {
        AdaptiveLayer {
            neurons: (0..initial_neurons).map(|_| AdaptiveNeuron::new(initial_neurons)).collect(),
            max_neurons,
            min_neurons,
            adaptation_threshold,
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.neurons.iter_mut().map(|neuron| neuron.activate(input)).collect()
    }

    pub fn backward(&mut self, error: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut next_error = vec![0.0; self.neurons[0].weights.len()];
        for (neuron, &neuron_error) in self.neurons.iter_mut().zip(error.iter()) {
            let gradients = neuron.calculate_gradients(neuron_error);
            neuron.update_weights(&gradients, learning_rate);
            for (i, &gradient) in gradients.iter().enumerate() {
                next_error[i] += gradient;
            }
        }
        next_error
    }

    pub fn adapt(&mut self, emotional_state: f64) {
        let mut rng = rand::thread_rng();

        for neuron in &mut self.neurons {
            neuron.update_importance(emotional_state);
        }

        self.neurons.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());

        if emotional_state > self.adaptation_threshold && self.neurons.len() < self.max_neurons {
            self.neurons.push(AdaptiveNeuron::new(self.neurons[0].weights.len()));
        } else if emotional_state < self.adaptation_threshold && self.neurons.len() > self.min_neurons {
            self.neurons.pop();
        }

        for neuron in &mut self.neurons {
            if rng.gen::<f64>() < 0.1 {
                neuron.mutate();
            }
        }
    }
}

impl AdaptiveNeuron {
    fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        AdaptiveNeuron {
            weights: (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            activation_history: VecDeque::with_capacity(100),
            importance_score: 0.0,
        }
    }

    fn activate(&mut self, input: &[f64]) -> f64 {
        let weighted_sum: f64 = input.iter().zip(self.weights.iter()).map(|(&x, &w)| x * w).sum();
        let activation = 1.0 / (1.0 + (-weighted_sum).exp());
        if self.activation_history.len() >= 100 {
            self.activation_history.pop_front();
        }
        self.activation_history.push_back(activation);
        activation
    }

    fn calculate_gradients(&self, error: f64) -> Vec<f64> {
        let last_activation = *self.activation_history.back().unwrap();
        let gradient = error * last_activation * (1.0 - last_activation);
        self.weights.iter().map(|&w| gradient * w).collect()
    }

    fn update_weights(&mut self, gradients: &[f64], learning_rate: f64) {
        for (weight, &gradient) in self.weights.iter_mut().zip(gradients.iter()) {
            *weight -= learning_rate * gradient;
        }
    }

    fn update_importance(&mut self, emotional_state: f64) {
        let avg_activation = self.activation_history.iter().sum::<f64>() / self.activation_history.len() as f64;
        self.importance_score = avg_activation * (1.0 - emotional_state);
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        for weight in &mut self.weights {
            if rng.gen::<f64>() < 0.1 {
                *weight += rng.gen_range(-0.1..0.1);
            }
        }
    }
}
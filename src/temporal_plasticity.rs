use rand::Rng;


pub struct TemporalNeuron {
    weights: Vec<f64>,
    delays: Vec<f64>,
    activation_history: Vec<(f64, f64)>, // (time, activation)
    plasticity: f64,
}

impl TemporalNeuron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        TemporalNeuron {
            weights: (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            delays: (0..input_size).map(|_| rng.gen_range(0.0..1.0)).collect(),
            activation_history: Vec::new(),
            plasticity: rng.gen_range(0.0..0.1),
        }
    }

    pub fn activate(&mut self, input: &[f64], time: f64) -> f64 {
        let weighted_sum: f64 = input.iter()
            .zip(self.weights.iter())
            .zip(self.delays.iter())
            .map(|((&x, &w), &d)| x * w * self.temporal_kernel(time - d))
            .sum();
        
        let activation = self.activation_function(weighted_sum);
        self.activation_history.push((time, activation));
        
        if self.activation_history.len() > 100 {
            self.activation_history.remove(0);
        }
        
        activation
    }

    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    pub fn calculate_gradients(&self, error: f64) -> Vec<f64> {
        let (time, last_activation) = self.activation_history.last().unwrap();
        let gradient = error * self.activation_function_derivative(last_activation);
        
        self.weights.iter()
            .zip(self.delays.iter())
            .map(|(&w, &d)| gradient * w * self.temporal_kernel(*time - d))
            .collect()
    }

    pub fn update_weights(&mut self, gradients: &[f64], learning_rate: f64) {
        for ((weight, delay), &gradient) in self.weights.iter_mut()
            .zip(self.delays.iter_mut())
            .zip(gradients.iter()) {
            *weight -= learning_rate * gradient;
            *delay -= learning_rate * self.plasticity * gradient;
            *delay = delay.clamp(0.0, 1.0); // Ensure delay stays in [0, 1]
        }
    }

    fn temporal_kernel(&self, t: f64) -> f64 {
        // Using a simple exponential decay kernel
        (-t.abs()).exp()
    }

    fn activation_function(&self, x: f64) -> f64 {
        // Sigmoid activation function
        1.0 / (1.0 + (-x).exp())
    }

    fn activation_function_derivative(&self, y: &f64) -> f64 {
        // Derivative of sigmoid function
        y * (1.0 - y)
    }
}

pub struct TemporalLayer {
    pub neurons: Vec<TemporalNeuron>,
}

impl TemporalLayer {
    pub fn new(size: usize) -> Self {
        TemporalLayer {
            neurons: (0..size).map(|_| TemporalNeuron::new(size)).collect(),
        }
    }

    pub fn forward(&mut self, input: &[f64], time: f64) -> Vec<f64> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.activate(input, time))
            .collect()
    }

    pub fn backward(&mut self, error: &[f64], learning_rate: f64) -> Vec<f64> {
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
use std::collections::HashMap;

pub struct NeuroSymbolicLayer {
    symbolic_rules: HashMap<String, Box<dyn Fn(&[f64]) -> f64>>,
    neural_output: Vec<f64>,
}

impl NeuroSymbolicLayer {
    pub fn new() -> Self {
        NeuroSymbolicLayer {
            symbolic_rules: HashMap::new(),
            neural_output: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, name: &str, rule: Box<dyn Fn(&[f64]) -> f64>) {
        self.symbolic_rules.insert(name.to_string(), rule);
    }

    pub fn process(&mut self, mut input: Vec<f64>) -> Vec<f64> {
        self.neural_output = input.clone();
        
        for rule in self.symbolic_rules.values() {
            let symbolic_output = rule(&input);
            input.push(symbolic_output);
        }
        
        input
    }

    pub fn backward(&self, error: &[f64]) -> Vec<f64> {
        let mut neural_error = vec![0.0; self.neural_output.len()];
        let mut symbolic_gradients = HashMap::new();

        // Calculate gradients for symbolic rules
        for (name, rule) in &self.symbolic_rules {
            let mut gradient = vec![0.0; self.neural_output.len()];
            let epsilon = 1e-5;

            for i in 0..self.neural_output.len() {
                let mut pos_input = self.neural_output.clone();
                let mut neg_input = self.neural_output.clone();
                pos_input[i] += epsilon;
                neg_input[i] -= epsilon;

                let pos_output = rule(&pos_input);
                let neg_output = rule(&neg_input);
                gradient[i] = (pos_output - neg_output) / (2.0 * epsilon);
            }

            symbolic_gradients.insert(name.clone(), gradient);
        }

        // Combine gradients from neural and symbolic parts
        for i in 0..self.neural_output.len() {
            neural_error[i] += error[i]; // Direct error from neural part

            // Add contributions from symbolic rules
            for (name, gradient) in &symbolic_gradients {
                let rule_index = self.neural_output.len() + self.symbolic_rules.keys().position(|k| k == name).unwrap();
                neural_error[i] += error[rule_index] * gradient[i];
            }
        }

        neural_error
    }

    pub fn explain(&self) -> Vec<String> {
        let mut explanations = Vec::new();
        for (name, rule) in &self.symbolic_rules {
            let output = rule(&self.neural_output);
            explanations.push(format!("Rule '{}' output: {:.2}", name, output));
        }
        explanations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuro_symbolic_layer() {
        let mut layer = NeuroSymbolicLayer::new();
        
        // Add a simple rule: sum of inputs
        layer.add_rule("sum", Box::new(|inputs: &[f64]| inputs.iter().sum()));
        
        // Process some input
        let input = vec![1.0, 2.0, 3.0];
        let output = layer.process(input);
        
        // Check output
        assert_eq!(output, vec![1.0, 2.0, 3.0, 6.0]);
        
        // Test backward pass
        let error = vec![0.1, 0.2, 0.3, 0.4];
        let gradients = layer.backward(&error);
        
        // Check gradients (should be original error plus contributions from symbolic rule)
        assert_eq!(gradients.len(), 3);
        assert!(gradients.iter().all(|&g| g > 0.0));
        
        // Test explanations
        let explanations = layer.explain();
        assert_eq!(explanations, vec!["Rule 'sum' output: 6.00"]);
    }
}
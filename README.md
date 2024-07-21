# Neuroforge
## 1. Usage
```
use neuroforge::{NeuroForge, neuro_symbolic::NeuroSymbolicLayer};

fn main() {
    let mut network = NeuroForge::new(&[5, 10, 3]);
    let mut symbolic_layer = NeuroSymbolicLayer::new();

    symbolic_layer.add_rule("high_activation", Box::new(|output: &[f64]| {
        if output.iter().any(|&x| x > 0.8) { 1.0 } else { 0.0 }
    }));

    // Train the network
    let inputs = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]];
    let targets = vec![vec![0.9, 0.1, 0.5]];
    network.train(&inputs, &targets, 1000);

    // Make a prediction
    let input = vec![0.2, 0.3, 0.4, 0.5, 0.6];
    let mut output = network.forward(&input);
    
    // Apply neuro-symbolic processing
    output = symbolic_layer.process(output);

    println!("Output: {:?}", output);
    println!("Explanations: {:?}", symbolic_layer.explain());
}
```
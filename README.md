# Neuroforge

Neuroforge is a Rust library for creating and training neural networks with an integrated neuro-symbolic processing layer.

## 1. Usage

Here's a basic example of how to use the `neuroforge` library:

```rust
use neuroforge::{NeuroForge, neuro_symbolic::NeuroSymbolicLayer};

fn main() {
    // Create a new neural network with 5 input neurons, 10 hidden neurons, and 3 output neurons
    let mut network = NeuroForge::new(&[5, 10, 3]);
    let mut symbolic_layer = NeuroSymbolicLayer::new();

    // Add a neuro-symbolic rule
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

## 2. Features
- Neural Network Creation: Easily create neural networks with a specified architecture.
- Training: Train the network with input-target pairs.
- Forward Propagation: Perform forward propagation to get network outputs.
- Neuro-Symbolic Processing: Integrate symbolic rules to post-process neural network outputs.

## 3. Installation
Add `neuroforge` to your `Cargo.toml`:
```toml
[dependencies]
neuroforge = "*"
```
Or command:
```bash
cargo add neuroforge
```

## 4. License

This project is licensed under the MIT License.




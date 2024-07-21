// emotional_memory.rs

use std::collections::VecDeque;

pub struct EmotionalMemory {
    memories: VecDeque<(Vec<f64>, f64)>, // (memory, emotional_intensity)
    capacity: usize,
}

impl EmotionalMemory {
    pub fn new(capacity: usize) -> Self {
        EmotionalMemory {
            memories: VecDeque::new(),
            capacity,
        }
    }

    pub fn store(&mut self, memory: Vec<f64>, emotional_intensity: f64) {
        if self.memories.len() >= self.capacity {
            self.memories.pop_front();
        }
        self.memories.push_back((memory, emotional_intensity));
    }

    pub fn recall(&self, current_emotion: f64) -> Option<Vec<f64>> {
        self.memories
            .iter()
            .max_by(|a, b| {
                let a_relevance = (current_emotion - a.1).abs();
                let b_relevance = (current_emotion - b.1).abs();
                a_relevance.partial_cmp(&b_relevance).unwrap()
            })
            .map(|(memory, _)| memory.clone())
    }
}
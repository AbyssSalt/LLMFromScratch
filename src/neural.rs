use crate::tokenizer;
use crate::tensors::{Tensor, activation_func, add_tensors, mult_tensors, transpose};
use rand::{self, RngExt};
use std::{collections::HashMap, env::var};



pub fn softmax(data: &mut Tensor) {
    for y in 0..data.sizes[1] {
        let start: usize = y * data.strides[1];
        let end: usize = start + data.strides[1];
        
        let max_val: f32 = data.data[start..end].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum: f32 = 0.0;
        for i in start..end {
            data.data[i] = (data.data[i] - max_val).exp();
            sum += data.data[i];
        }
        for i in start..end {
            data.data[i] /= sum;
        }
    }
}

pub fn ffn(data: &Tensor, weight_1: &Tensor, bias_1: &Tensor, weight_2: &Tensor, bias_2: &Tensor) -> Tensor {
    let first_step: Tensor = add_tensors(&mult_tensors(data, weight_1).expect("failed multiplying"), bias_1).expect("failed adding");
    let second_step: Tensor = activation_func(&first_step);
    let third_step: Tensor = add_tensors(&mult_tensors(&second_step, &weight_2).expect("failed multiplying"), bias_2).expect("failed adding");
    return third_step;
}

pub fn qkv(data: &Tensor, weight_q: &Tensor, weight_k: &Tensor, weight_v: &Tensor) -> Tensor {
    let mut q: Tensor = mult_tensors(data, weight_q).expect("Failed to compute q");
    let mut k: Tensor = mult_tensors(data, weight_k).expect("Failed to compute k");
    let mut v: Tensor = mult_tensors(data, weight_v).expect("Failed to compute v");

    transpose(&mut k);

    let mut scores: Tensor = mult_tensors(&q, &k).expect("Failed to calculate scores");
    let scale: f32 = 1.0 / (weight_q.sizes[0] as f32).sqrt(); // Assuming weight_q.sizes[0] is head_dim
    for i in 0..scores.data.len() {
        scores.data[i] *= scale;
    }
    for y in 0..scores.sizes[1] {
        for x in 0..scores.sizes[0] {
            if x > y {
                scores.data[x + y * scores.strides[1]] = -f32::MAX;
            }
        }
    }

    softmax(&mut scores);
    let attention_output: Tensor = mult_tensors(&scores, &v).expect("Failed final attention multiply");
    return attention_output;
}

pub fn layer_norm(data: &Tensor) -> Tensor {
    let mut out: Tensor = Tensor { data: (vec![0.0; data.strides[3] * data.sizes[0]]), sizes: (data.sizes), strides: (data.strides) };

    for y in 0..out.sizes[1] {
        let mut mean: f32 = 0.0;
        for x in 0..out.sizes[0] {
            mean += data.data[data.strides[1] * y + x];
        }
        mean /= data.strides[1] as f32;

        let mut variance: f32 = 0.0;
        for x in 0..out.sizes[0] {
            variance += (data.data[data.strides[1] * y + x] - mean).powi(2);
        }
        variance /= data.strides[1] as f32;
        variance += f32::MIN_POSITIVE;
        variance = variance.sqrt();

        for x in 0..out.sizes[0] {
            out.data[data.strides[1] * y + x] = (data.data[data.strides[1] * y + x] - mean) / variance;
        }
    }
    
    return out;
}

pub fn transformer_block(x: &Tensor, w_q: &Tensor, w_k: &Tensor, w_v: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor) -> Tensor {
    let x_norm = layer_norm(x);
    let attn_out = qkv(&x_norm, w_q, w_k, w_v);
    let x_attn = add_tensors(x, &attn_out).expect("Residual 1 failed");

    let x_attn_norm = layer_norm(&x_attn);
    let ffn_out = ffn(&x_attn_norm, w1, b1, w2, b2);
    let y = add_tensors(&x_attn, &ffn_out).expect("Residual 2 failed");

    return y
}

pub fn generate_weight_tensor(size: [usize; 4]) -> Tensor {
    let out_strides: [usize; 4] = [1, size[0], size[1] * size[0], size[2] * size[1] * size[0]];
    let mut rng: rand::prelude::ThreadRng = rand::rng();
    let mut out: Tensor = Tensor { data: (vec![0.0; out_strides[3] * size[3]]), sizes: (size), strides: (out_strides) };
    let scale = (1.0 / (size[0] as f32)).sqrt();
    
    for i in 0..out.data.len() {
        out.data[i] = (rng.random::<f32>() - 0.5) * 2.0 * scale;
    }

    return out;
}

pub fn generate_embedding(vocabulary: &HashMap<char, u8>, embed_dim: usize) -> Tensor {
    let char_count: usize = vocabulary.keys().len();
    let mut rng: rand::prelude::ThreadRng = rand::rng();
    
    let mut out: Tensor = Tensor { data: (vec![0.0; char_count * embed_dim]), sizes: ([embed_dim, char_count, 1, 1]), strides: ([1, embed_dim, embed_dim * char_count, embed_dim * char_count]) };
    let scale = (1.0 / (embed_dim as f32)).sqrt();

    for i in 0..out.data.len() {
        out.data[i] = (rng.random::<f32>() - 0.5) * 2.0 * scale;
    }

    return out;
}

pub fn embed_text(ids: &Vec<u8>, embedding_tensor: &Tensor) -> Tensor {
    let out_sizes: [usize; 4] = [embedding_tensor.sizes[0], ids.len(), 1, 1];
    let out_strides: [usize; 4] = [1, out_sizes[0], out_sizes[1] * out_sizes[0], out_sizes[1] * out_sizes[0]];
    let mut out: Tensor = Tensor { data: (Vec::with_capacity(ids.len() * embedding_tensor.sizes[0])), sizes: (out_sizes), strides: (out_strides) };

    for &id in ids {
        let start = id as usize * embedding_tensor.strides[1];
        let end = start as usize + embedding_tensor.strides[1];

        out.data.extend_from_slice(&embedding_tensor.data[start..end]);
    }

    return out
}


// What the hell is any of this even doing ;-;
pub fn forward_and_sample(input_ids: &Vec<u8>, embedding_tensor: &Tensor, w_q: &Tensor, w_k: &Tensor, w_v: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor, lm_head_weight: &Tensor) -> u8 {
    // 1. Embed input sequence tokens
    let x = embed_text(input_ids, embedding_tensor);

    // 2. Complete Forward pass through your architecture block
    let processed = transformer_block(&x, w_q, w_k, w_v, w1, b1, w2, b2);

    // 3. Project hidden layer back to vocabulary dimensions (LM Head)
    // Result shape: (vocab_size, sequence_length)
    let logits = mult_tensors(&processed, lm_head_weight).expect("LM Head projection matrix multiplication failed");

    // 4. Extract the logits for the very last token in the sequence safely
    let vocab_size = lm_head_weight.sizes[0]; // should match your vocabulary length (4)
    
    let total_len = logits.data.len();
    let last_token_start = total_len - vocab_size;
    let last_token_end = total_len;

    let mut next_token_logits = Tensor {
        data: logits.data[last_token_start..last_token_end].to_vec(),
        sizes: [vocab_size, 1, 1, 1],
        strides: [1, vocab_size, vocab_size, vocab_size],
    };

    // 5. Convert raw scores to concrete choice probabilities
    softmax(&mut next_token_logits);

    // 6. Basic Weighted Random Sample Selection
    let mut rng = rand::rng();
    let roll: f32 = rng.random::<f32>(); // Pick random decimal 0.0..1.0
    let mut cumulative_prob = 0.0;
    
    for (id, &prob) in next_token_logits.data.iter().enumerate() {
        cumulative_prob += prob;
        if roll <= cumulative_prob {
            return id as u8;
        }
    }

    return (next_token_logits.data.len() - 1) as u8
}

pub fn main() {
    // 1. Setup a tiny example vocabulary (e.g., for "abc ")

    let maps: (HashMap<char, u8>, HashMap<u8, char>) = tokenizer::generate_vocabulary(&String::from("D:\\Visual Studio\\Rust\\LLM\\Data Parser\\CleanedData\\Hestle.txt"));

    let mut vocabulary = maps.0;
    let embed_dim = 64;
    let vocab_size = vocabulary.len();

    // 2. Initialize all network weights using your custom functions
    println!("Initializing model weights...");
    let embedding_table = generate_embedding(&vocabulary, embed_dim);
    
    // Attention weights
    let w_q = generate_weight_tensor([embed_dim, embed_dim, 1, 1]);
    let w_k = generate_weight_tensor([embed_dim, embed_dim, 1, 1]);
    let w_v = generate_weight_tensor([embed_dim, embed_dim, 1, 1]);


    // FFN weights (Up-projection expands by 4x)
    let seq_len = 3; // This is 3 for your seed prompt
    let w1 = generate_weight_tensor([4 * embed_dim, embed_dim, 1, 1]); 
    let w2 = generate_weight_tensor([embed_dim, 4 * embed_dim, 1, 1]);
    let b1 = generate_weight_tensor([4 * embed_dim, seq_len, 1, 1]);
    let b2 = generate_weight_tensor([embed_dim, seq_len, 1, 1]);

    // LM Head Matrix (Projects back from embed_dim to vocab_size)
    let lm_head_weight = generate_weight_tensor([vocab_size, embed_dim, 1, 1]);

    // 3. Define a starting seed prompt (e.g., "abc") mapped to IDs
    // "a" -> 0, "b" -> 1, "c" -> 2
    let mut input_ids: Vec<u8> = vec![0, 1, 2];
    
    // Create a reverse mapping so we can print the generated tokens back as text
    let id_to_char: std::collections::HashMap<u8, char> = vocabulary
        .iter()
        .map(|(&c, &id)| (id, c))
        .collect();

    print!("Seed prompt: ");
    for id in &input_ids {
        print!("{}", id_to_char.get(id).unwrap_or(&'?'));
    }
    println!("\nGenerating...");

    // 4. Autoregressive Generation Loop
    // The model predicts 1 token, appends it to the prompt, and runs again
    for _ in 0..80 {
        let next_id = forward_and_sample(
            &input_ids,
            &embedding_table,
            &w_q, &w_k, &w_v,
            &w1, &b1, &w2, &b2,
            &lm_head_weight,
        );

        // Convert the predicted ID back to a character and print it live
        let next_char = id_to_char.get(&next_id).unwrap_or(&'?');
        print!("{}", next_char);
        
        // Ensure stdout flushes immediately so text appears character-by-character
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        // Append the new ID back into the sequence to use as context for the next step
        input_ids.push(next_id);
    }
    println!("\n\nGeneration complete!");
}
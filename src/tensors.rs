struct Tensor {
    data: Vec<f32>,
    // x, y, z, w
    sizes: [usize;4],
    strides: [usize;4],
}

fn mult_matrices(a: &Tensor, b: &Tensor, out: &mut Tensor, a_offset: usize, b_offset: usize, out_offset: usize) {
    // AxB results in an Ay by Bx matrix
    
    // iterate through each row of A
    for y in 0..a.sizes[1] {

        // iterate for Matrix Multiplication (column of a, row of b)
        for i in 0..a.sizes[0] {
                let a_index: usize = a_offset + y * a.strides[1] + i * a.strides[0];
                let a_value: f32 = a.data[a_index];
                
                // iterate through each column of B
                for x in 0..b.sizes[0] {
                    let b_index: usize = b_offset + x * b.strides[0] + i * b.strides[1];
                    let out_index: usize = out_offset + x * out.strides[0] + y * out.strides[1];
                    
                    // By doing this odd ordering we end up incrementing A and B's indices by 1 allowing for better cpu usage
                    out.data[out_index] += a_value * b.data[b_index];
                }
            }

    }
}

fn mult_tensors(a: &Tensor, b: &Tensor) -> Option<Tensor> {
    if a.sizes[0] != b.sizes[1] || a.sizes[2] != b.sizes[2] || a.sizes[3] != b.sizes[3] {
        return None
    }

    let out_sizes: [usize; 4] = [b.sizes[0], a.sizes[1], a.sizes[2], a.sizes[3]];
    let out_strides: [usize; 4] = [1, out_sizes[0], out_sizes[1] * out_sizes[0], out_sizes[2] * out_sizes[1] * out_sizes[0]];
    let mut out: Tensor = Tensor { data: vec![0.0; out_strides[3] * out_sizes[3]], sizes: (out_sizes), strides: (out_strides) };

    for d3 in 0..a.sizes[2] {
        for d4 in 0..a.sizes[3] {
            let a_offset: usize = d3 * a.strides[2] + d4 * a.strides[3];
            let b_offset: usize = d3 * b.strides[2] + d4 * b.strides[3];
            let out_offset: usize = d3 * out.strides[2] + d4 * out.strides[3];
            mult_matrices(&a, &b, &mut out, a_offset, b_offset, out_offset);
        }
    }
    
    return Some(out);
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Option<Tensor> {
    if a.sizes != b.sizes {
        return None
    }
    let mut out: Tensor = Tensor { data: (vec![0.0; a.data.len()]), sizes: (a.sizes), strides: (a.strides) };
    for i in 0..a.data.len(){
        out.data[i] = a.data[i] + b.data[i];
    }
    return Some(out);
}

fn scale_tensor(a: &Tensor, b: f32) -> Tensor {
    let mut out: Tensor = Tensor { data: (vec![0.0; a.data.len()]), sizes: (a.sizes), strides: (a.strides) };
    for i in 0..a.data.len(){
        out.data[i] = a.data[i] * b;
    }
    return out;
}

fn activation_func(a: &Tensor) -> Tensor {
    let mut out: Tensor = Tensor { data: (vec![0.0; a.data.len()]), sizes: (a.sizes), strides: (a.strides) };
    for i in 0..a.data.len(){
        out.data[i] = (a.data[i]).max(0.0);
    }
    return out
}

pub fn main() {
    let mut a: Tensor = Tensor { data: (Vec::new()), sizes: ([2, 2, 2, 2]), strides: ([1, 2, 4, 8]) };
    a.data.append(&mut Vec::from([5.0, 1.0, 9.0, 3.0, 1.0, 2.0, 5.0, 2.0, 2.0, 7.0, 9.0, 2.0, 8.0, 5.0, 4.0, 9.0]));
    let mut b: Tensor = Tensor { data: (Vec::new()), sizes: ([1, 2, 2, 2]), strides: ([1, 1, 2, 4]) };
    b.data.append(&mut Vec::from([0.0, 1.0, 8.0, 8.0, 1.0, 4.0, 7.0, 7.0]));

    let c: Tensor = mult_tensors(&a, &b);
    // 1, 3 | ?, ? | 30, 17 | 91, 91
    println!("{:?}", c.data);
}
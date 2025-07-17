// Attention mechanism
use crate::{autodiff, Module, Tensor};
use crate::nn::Linear;

use super::{LayerNorm, SeqF, Sequential};

// ======================= Attention ======================= 

#[allow(non_snake_case)]
pub fn Attention (Q:Tensor, K:Tensor, V:Tensor, d_model:usize) -> Tensor {
    autodiff::dot(
        (
            autodiff::dot(Q, K.t()) / 
            (d_model as f32).sqrt()
        ).softmax(-1),
        V
    )
}

#[allow(non_snake_case)]
pub fn AttentionMasked (Q:Tensor, K:Tensor, V:Tensor, d_model:usize, mask:Tensor) -> Tensor {
    autodiff::dot(
        (
            autodiff::dot(Q, K.t()) / 
            (d_model as f32).sqrt() * 
            mask
        ).softmax(-1),
        V
    )
}

// ======================= MultiHead Attention =======================

pub struct MultiHeadAttention {
    pub wq: Vec<Linear>,
    pub wk: Vec<Linear>,
    pub wv: Vec<Linear>,
    pub wo: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub mask: Option<Tensor>
}

impl Module for MultiHeadAttention {
    fn params (&self) -> Vec<Tensor> {
        vec![
            self.wq.iter().map(|i| i.params()).collect::<Vec<Vec<Tensor>>>().concat(),
            self.wk.iter().map(|i| i.params()).collect::<Vec<Vec<Tensor>>>().concat(),
            self.wv.iter().map(|i| i.params()).collect::<Vec<Vec<Tensor>>>().concat(),
            self.wo.params()
        ].concat()
    }
}

impl MultiHeadAttention {
    pub fn f (&self, query: Tensor, key: Tensor, value: Tensor) -> Tensor {
        let head_out: Vec<Tensor> = if let Some(m) = &self.mask {
            // masked attention
            (0..self.num_heads).map(|i|
                AttentionMasked(
                    self.wq[i].f(query.clone()),
                    self.wk[i].f(key.clone()),
                    self.wv[i].f(value.clone()),
                    self.d_model,
                    m.clone()
                )
            ).collect()
        } else {
            // no masked attention
            (0..self.num_heads).map(|i|
                Attention(
                    self.wq[i].f(query.clone()),
                    self.wk[i].f(key.clone()),
                    self.wv[i].f(value.clone()),
                    self.d_model
                )
            ).collect()
        };

        self.wo.f(autodiff::concat(head_out, -1))
    }
}

#[allow(non_snake_case)]
pub fn MultiHeadAttention (d_model: usize, num_heads: usize) -> MultiHeadAttention {
    assert!(d_model % num_heads == 0, "d_model must be divisible by num heads");
    let d_v = d_model / num_heads;

    MultiHeadAttention {
        wq: (0..num_heads).map(|_| Linear(d_model, d_v, false)).collect(),
        wk: (0..num_heads).map(|_| Linear(d_model, d_v, false)).collect(),
        wv: (0..num_heads).map(|_| Linear(d_model, d_v, false)).collect(),
        wo: Linear(d_model, d_model, false),
        num_heads,
        d_model,
        mask: None
    }
}

#[allow(non_snake_case)]
pub fn MaskedMultiHeadAttention (d_model: usize, num_heads: usize, mask: Tensor) -> MultiHeadAttention {
    assert!(d_model % num_heads == 0, "d_model must be divisible by num heads");
    let d_v = d_model / num_heads;

    MultiHeadAttention {
        wq: (0..num_heads).map(|_| Linear(d_model, d_v, false)).collect(),
        wk: (0..num_heads).map(|_| Linear(d_model, d_v, false)).collect(),
        wv: (0..num_heads).map(|_| Linear(d_model, d_v, false)).collect(),
        wo: Linear(d_model, d_model, false),
        num_heads,
        d_model,
        mask: Some(mask)
    }
}
// ======================= Attention Feedforward =======================
pub struct AttentionFeedforward {
    w_expand: Linear,   // d_model to inner_dim
    w_contract: Linear, // inner_dim to d_model
}

#[allow(non_snake_case)]
pub fn AttentionFeedforward (d_model:usize, inner_dim: usize) -> AttentionFeedforward {
    AttentionFeedforward {
        w_expand: Linear(d_model, inner_dim, true),
        w_contract: Linear(inner_dim, d_model, true),
    }
}

impl Module for AttentionFeedforward {
    fn params (&self) -> Vec<Tensor> {
        vec![self.w_expand.params(), self.w_contract.params()].concat()
    }
}

impl SeqF for AttentionFeedforward {
    fn f (&self, x: Tensor) -> Tensor {
        self.w_contract.f(self.w_expand.f(x))
    }
}

// ======================= Transformer Encoder Layer =======================
pub struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    layer_norm_att: LayerNorm, // TODO: better naming!
    layer_norm_ffwd: LayerNorm,
    ffwd: AttentionFeedforward
}

#[allow(non_snake_case)]
pub fn TransformerEncoderLayer (d_model: usize, num_heads: usize, ff_dim: usize) -> TransformerEncoderLayer {
    TransformerEncoderLayer {
        attention: MultiHeadAttention(d_model, num_heads),
        layer_norm_att: LayerNorm(d_model),
        layer_norm_ffwd: LayerNorm(d_model),
        ffwd: AttentionFeedforward(d_model, ff_dim)
    }
}

impl Module for TransformerEncoderLayer {
    fn params (&self) -> Vec<Tensor> {
        vec![self.attention.params(), self.ffwd.params(), self.layer_norm_att.params(), self.layer_norm_ffwd.params()].concat()
    }
}

impl SeqF for TransformerEncoderLayer {
    fn f (&self, x: Tensor) -> Tensor {
        let to_add = x.clone();
        let y = self.attention.f(x.clone(), x.clone(), x.clone());
        let y = self.layer_norm_att.f(to_add + y);

        let to_add = y.clone();
        let y = self.ffwd.f(y);
        let y = self.layer_norm_ffwd.f(to_add + y);

        y
    }
}

// ======================= Transformer Encoder =======================
pub struct TransformerEncoder {
    layers: Sequential
}

#[allow(non_snake_case)]
pub fn TransformerEncoder (num_layers: usize, d_model: usize, num_heads: usize, ff_dim: usize) -> TransformerEncoder {
    let mut seq = Sequential();

    for _ in 0..num_layers {
        seq.insert(TransformerEncoderLayer(d_model, num_heads, ff_dim));
    }

    TransformerEncoder { layers: seq }     
}

impl Module for TransformerEncoder {
    fn params (&self) -> Vec<Tensor> {
        self.layers.params()
    }
}

impl SeqF for TransformerEncoder {
    fn f (&self, x: Tensor) -> Tensor {
        self.layers.f(x)
    }
}
---
title: 关于Seq2Seq的增量解码
mathjax: true
date: 2022-02-21 23:25:19
tags:
  - NLG
  - Decoder
  - 加速
---

这几天重构了一版生成模型。相较于各种各样的nlu任务、自然语言生成在工业中的应用并不多，一是由于实际场景下往往需要目标文本“受控”，二是生成过程自回归，无论有多大算力，仍然需要一个个token往外蹦。encoder-decoder架构下的自回归语言模型，如果不做任何优化，整个序列的计算复杂度是长度的三次项。本文旨在讨论解码过程中的性能优化。阅读本文，你需要知道seq2seq模型、了解attention结构、听说过bert/bart，并且最好熟悉tf1.x框架。文中不含开箱即跑的脚本/代码，但是如果你正在寻求加快生成的方法，那么文中的代码片段或许对你有参考价值。

## 模型结构

seq2seq的模型结构参考bart，bert的后一半取出做decoder，参数量与bert-base相当，6层encoder，6层decoder。

![模型结构](/post_images/incremental-decoder__model.drawio.svg)

训练任务以seq2seq的方式组织。

## 一些背景

语言模型大致分为Denosing（MLM）与AutoRegressive（AR）两类。Denosing以bert为代表，整个输入对模型完全可见，上下文信息完备，因此在nlu任务上有天然优势；AR以T5、GPT等为代表，预训练时模型只能看到前序文本，因此在nlg任务上表现更好。现阶段的语言模型，几乎都包括MultiHeadAttention(MHA)结构。
$$
Att(q,k,v)=softmax(\frac{QK^T}{\sqrt{d}})V
$$
$$
Q=W_qq
$$
$$
K=W_kk
$$
$$
V=W_vv
$$

“MultiHead”没有体现在公式上，实现中添加几个reshape就能做到多头的效果。如果qkv相同，MHA就叫做self-attention，常用在Encoder部分；如果q与kv来自两个输入，MHA叫做cross-attention，常用在Decoder部分。

MHA的实现可以参考[google-research/bert](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L558)。

## Encoder加速

encoder部分的输出在解码过程中始终不变，缓存输出，避免重复计算即可。

## Decoder加速

为了方便描述，约定第$n$个token的向量表示为$X_n$，序列为$Seq(n)={X_0,X_1,...,X_n}$，来看decoder中各个结构的缓存方案。

### FFN与Add & LayerNorm

FFN层没有token之间的交互，nlp的norm层是在最后一维上计算，这两层在长度这一维上拼接新token输出与缓存就能实现增量计算，计算量由$O(n)$降为$O(1)$。
### Decoder Self-Attention

![incremental-decoder__self-att-mask.drawio](/post_images/incremental-decoder__self-att-mask.drawio.svg)

decoder的self-attention部分，训练中计算att-score时会按照下三角的方式mask，第$n$个token只与前$n-1$个token及自己交互，以防止后续token的“泄漏”。实际上解码第$n$个token时，有效计算只有$Att(X_n,Seq(n),Seq(n))$（图中的紫色行）。前$n-1$个token对$n$及$n$以后的att-score始终为0（图中白色列）。

![incremental-decoder__self-att-cal.drawio](/post_images/incremental-decoder__self-att-cal.drawio.svg)

实现这一步的缓存，self-attention需要的输入是：

- 第$n$个新token的向量表示（$q$）
- 第$n-1$步的att-score
- 第$n-1$步的$W_k(Seq(n-1))$
- 第$n-1$步的$W_v(Seq(n-1))$

输出是：

- att结果

- 第$n$ 步的att-score
- 第$n$步的$W_k(Seq(n))$
- 第$n$步的$W_v(Seq(n))$

### Decoder Cross-Attention

![incremental-decoder__self-att-mask.drawio](/post_images/incremental-decoder__cross-att-mask.drawio.svg)

decoder的cross-attention部分，训练时att-score会有正方形的mask。不同于下三角，正方形mask下每个新token都会参与计算，而softmax无法增量获取结果，因此解码过程中无法向self-attention那样减少att-score的计算，只能缓存到qkv映射这一步。其中q是encoder输出，始终不变，kv的线性变换可以通过缓存的方式节省计算。实现带缓存的cross-attention需要的输入是：

- $W_q(EncoderOut)$
- 第$n-1$步的$W_k(Seq(n-1))$
- 第$n-1$步的$W_v(Seq(n-1))$

输出是：

- att结果
- 第$n$步的$W_k(Seq(n))$
- 第$n$步的$W_v(Seq(n))$

## 计算量

[《线性Transformer应该不是你要的模型》][1]一文中有对计算量的评估，假设$n$为序列长度，$d$为head_size（64），$h$为head的数目（12），$hd$为常说的 hidden_size（768）。我们来看看缓存下的增量解码理论上的计算量评估。

### FFN

FFN层比较简单，无缓存时第一层是$n×hd$的矩阵乘以$hd×4hd$的矩阵，第二层是$n×4hd$的矩阵乘以$4hd×hd$的矩阵的计算，计算量为$8n(hd)^2$缓存后降至$8(hd^2)$

### Decoder Self-Attention

无缓存时，援引文中的计算量评估：

> 对于SA来说，一开始是$Q$,$K$,$V$的投影变换，即$n×hd$的矩阵乘以$hd×hd$的矩阵做3次，因此计算量是$3n(hd)^2$；然后是$h$个Attention头的运算，每个头先是$n×d$的$Q$与$d×n$的$K^T$相乘得到$n×n$的Attention矩阵（softmax和归一化的计算量暂且忽略），然后$n×n$的矩阵与$n×d$的$V$相乘得到$n×d$的矩阵，这两步的计算量都是$n^2d$，所以总计算量是$h(n^2d+n^2d)$；最后的输出还有一个投影变换，也是$n×hd$的矩阵乘以$hd×hd$的矩阵，计算量是$n(hd)^2$。所以，SA的总计算量是
> $$
> 3n(hd)^2+h(n^2d+n^2d)+n(hd)^2=4nh^2d^2+2n^2hd
> $$

缓存下，计算简化成：

$$Att(X_n,Seq(n),Seq(n))=softmax(\frac{(W_qX_n)(W_kSeq(n)))^T}{\sqrt{d}})W_vSeq(n)$$

相比于$Att(Seq(n),Seq(n),Seq(n))$，att-score的$O(n^2)$计算降为$O(n)$。另外，其中$W_kSeq(n)$可以由$W_kSeq(n-1)$与$W_kX_n$拼接得到，而$W_kSeq(n-1)$在之前的解码中计算过了，缓存$k、v$线性变换的结果，又可以将这一步的$O(n)$乘法降为$O(1)$。

在缓存下，$Q$,$K$,$V$的投影计算简化：

- $Q$为$1×hd$的矩阵乘以$hd×hd$的矩阵，计算量是$(hd)^2$
- $K$，$V$为$1×hd$乘$hd×hd$的矩阵，结果与$(n-1)×hd$缓存拼接，得到($n×hd)$的矩阵，计算量为$2(hd)^2$
- $h$个Attention头的运算，每个头先是$1×d$的$Q$与$d×n$的 $K^T$相乘得到$1×n$的Attention矩阵，然后$1×n$的矩阵与$n×d$的$V$相乘的道$1×d$的矩阵，计算量为$2nd$，总量为$2nhd$
- 最后输出的投影变换，是$1×hd$乘以权重$hd×hd$，计算量是$(hd)^2$

所以，使用缓存的self-attention总计算量是：
$$
(hd)^2+2(hd)^2+2nhd+(hd)^2=2nhd+4h^2d^2
$$
### Decoder Cross-Attention

无缓存的cross-attention计算量仍是$4nh^2d^2+2n^2hd$，使用缓存，计算量是：

- $Q$始终缓存，仅第一次有计算量，这里忽略
- $K$，$V$为$1×hd$乘$hd×hd$的矩阵，结果与$(n-1)×hd$缓存拼接，得到($n×hd)$的矩阵，计算量为$2(hd)^2$
- $h$个Attentioin头的运算仍是$h(n^2d+n^2d)$
- 输出投影变换仍是$n(hd^2)$

### Total

列个表来看有无缓存下，每个decoder模块的计算量比较

|                               | 无缓存             | 有缓存                          |
| ----------------------------- | ------------------ | ------------------------------- |
| 2个FFN层                      | $8n(hd)^2$         | $8(hd)^2$                       |
| self-att，$Q$投影变换         | $n(hd)^2$          | $(hd)^2$                        |
| self-att，$K$投影变换         | $n(hd)^2$          | $(hd)^2$                        |
| self-att，$V$投影变换         | $n(hd)^2$          | $(hd)^2$                        |
| self-att，attention-score计算 | $2n^2hd$           | $2nhd$                          |
| self-att，输出投影变换        | $n(hd)^2$          | $(hd^2)$                        |
| cross-att，$Q$投影变换        | $n(hd)^2$          | 0                               |
| self-att，$K$投影变换         | $n(hd)^2$          | $(hd)^2$                        |
| self-att，$V$投影变换         | $n(hd)^2$          | $(hd)^2$                        |
| self-att，attention-score计算 | $2n^2hd$           | $2n^2hd$                        |
| self-att，输出投影变换        | $n(hd)^2$          | $n(hd)^2$                       |
| 总共                          | $4n^2hd+16n(hd)^2$ | $2n^2hd+2nhd+2n(hd)^2+14(hd)^2$ |
| $h=12,d=64$                   | $3072n^2+9437184n$ | $1536n^2+1181184n+8257536$      |

下图展示了目标序列长度$n$与（缓存计算量）/（无缓存计算量）的关系，在长度为128时，计算量只有原来的14.85%（碰巧任务的目标序列长度在120左右）

![incremental-decoder__self-att-mask.drawio](/post_images/incremental-decoder__cal_ratio.png)

## Code

```python
import tensorflow as tf
from typing import *


def dropout(input_tensor: tf.Tensor, dropout_prob: float) -> tf.Tensor:
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

def layer_norm(input_tensor: tf.Tensor, name: str = None) -> tf.Tensor:
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def multi_head_attention(query: tf.Tensor,
                         key_value: Optional[tf.Tensor] = None,
                         attention_mask: Optional[tf.Tensor] = None,
                         position_bias: Optional[tf.Tensor] = None,
                         cache_key_value_states: Optional[tf.Tensor] = None,
                         hidden_size: int = 768,
                         attention_head_n: int = 12,
                         size_per_head: int = 64,
                         attention_probs_dropout_prob: float = 0.2,
                         query_act: Optional[str] = None,
                         key_act: Optional[str] = None,
                         value_act: Optional[str] = None,
                         large_negative_number: float = tf.float32.min,
                         get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev)) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    is_cross_attention = key_value is not None

    batch_size, q_len, q_hidden_size = get_shape_list(query, expected_rank=3)

    if is_cross_attention:
        batch_size, kv_len, kv_hidden_size = get_shape_list(key_value, expected_rank=3)
        assert q_hidden_size == kv_hidden_size == hidden_size
    else:
        assert q_hidden_size == hidden_size

    assert size_per_head * attention_head_n == hidden_size

    scale = size_per_head ** -0.5
    query_proj = tf.reshape(
        tf.layers.dense(tf.reshape(query, [batch_size * q_len, hidden_size]),
                        units=attention_head_n * size_per_head,
                        activation=query_act,
                        name="query",
                        kernel_initializer=get_new_initializer()),
        [batch_size, q_len, hidden_size]
    )
    query_proj *= scale

    # ======== [获取kv states] START ========
    if is_cross_attention:
        if cache_key_value_states is not None:
            # cross attention，有缓存
            key_proj = cache_key_value_states[0]
            value_proj = cache_key_value_states[1]
        else:
            # cross attention，无缓存
            key_proj = tf.reshape(
                tf.layers.dense(tf.reshape(key_value, [batch_size * kv_len, hidden_size]),
                                units=attention_head_n * size_per_head,
                                activation=key_act,
                                name="key",
                                kernel_initializer=get_new_initializer()),
                [batch_size, kv_len, hidden_size]
            )
            value_proj = tf.reshape(
                tf.layers.dense(tf.reshape(key_value, [batch_size, kv_len, hidden_size]),
                                units=attention_head_n * size_per_head,
                                activation=value_act,
                                name="value",
                                kernel_initializer=get_new_initializer()),
                [batch_size, kv_len, hidden_size]
            )
    else:
        kv_len = q_len
        # self attention，无缓存
        key_proj = tf.reshape(
            tf.layers.dense(tf.reshape(query, [batch_size * q_len, hidden_size]),
                            units=attention_head_n * size_per_head,
                            activation=key_act,
                            name="key",
                            kernel_initializer=get_new_initializer()),
            [batch_size, q_len, hidden_size]
        )
        value_proj = tf.reshape(
            tf.layers.dense(tf.reshape(query, [batch_size, q_len, hidden_size]),
                            units=attention_head_n * size_per_head,
                            activation=value_act,
                            name="value",
                            kernel_initializer=get_new_initializer()),
            [batch_size, q_len, hidden_size]
        )
        if cache_key_value_states is not None:
            # self attention，有缓存（仅在decoder的情况下可能用到）
            key_proj = tf.concat([cache_key_value_states[0], key_proj], axis=1)
            value_proj = tf.concat([cache_key_value_states[1], value_proj], axis=1)
            kv_len = get_shape_list(key_proj, expected_rank=3)[1]

    updated_cache_key_value_states = (key_proj, value_proj)

    # query_proj: [bs, q_len, H]  key_proj: [bs, kv_len, H]  value_prod: [bs, kv_len, H]
    # ======== [获取kv states] END ========

    # ======== [计算attention score] START ========
    def transpose_to_bnlh(tensor: tf.Tensor, bs: int, sl: int):
        return tf.transpose(tf.reshape(tensor, shape=[bs, sl, attention_head_n, size_per_head]), (0, 2, 1, 3))

    # b*n, q, h
    query_proj = tf.reshape(
        transpose_to_bnlh(query_proj, batch_size, q_len),
        shape=[batch_size * attention_head_n, q_len, size_per_head]
    )
    # b*n, kv, h
    key_proj = tf.reshape(
        transpose_to_bnlh(key_proj, batch_size, kv_len),
        shape=[batch_size * attention_head_n, kv_len, size_per_head]
    )
    value_proj = tf.reshape(
        transpose_to_bnlh(value_proj, batch_size, kv_len),
        shape=[batch_size * attention_head_n, kv_len, size_per_head]
    )

    # b*n, q, kv
    att_weights = tf.matmul(query_proj, key_proj, transpose_b=True)

    # attention_mask & position_bias
    att_weights = tf.reshape(att_weights, shape=[batch_size, attention_head_n, q_len, kv_len])

    # b, q_len, kv_len
    if position_bias is not None:
        tf.assert_rank(position_bias, rank=3, message=f"position bias shape should be [batch_size, q_len, kv_len]")
        position_bias = tf.expand_dims(position_bias, axis=1)
        att_weights += position_bias

    # b, q_len, kv_len
    if attention_mask is not None:
        tf.assert_rank(attention_mask, rank=3, message=f"attention mask shape should be [batch_size, q_len, kv_len]")
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = (1.0 - attention_mask) * large_negative_number
        att_weights += attention_mask

    att_weights = tf.reshape(att_weights, shape=[batch_size * attention_head_n, q_len, kv_len])
    att_probs = tf.nn.softmax(att_weights, -1)

    att_probs = dropout(att_probs, attention_probs_dropout_prob)
    # ======== [计算attention prob] END ========

    # b*n, q_len, h
    # return tf.reshape(att_probs, [batch_size, attention_head_n, q_len, kv_len]), att_probs, att_probs
    att_output = tf.matmul(att_probs, value_proj)
    att_output = tf.transpose(
        tf.reshape(att_output, shape=(batch_size, attention_head_n, q_len, size_per_head)),
        (0, 2, 1, 3)
    )
    att_output = tf.reshape(att_output, shape=[batch_size, q_len, attention_head_n * size_per_head])
    return (
        att_output,
        tf.reshape(att_weights, shape=(batch_size, attention_head_n, q_len, kv_len)),
        updated_cache_key_value_states
    )
```

这段mha代码可以替换google/bert中的attention，根据提供输入的不同，可以实现不同的attention：

|                       | query                     | key_value                                              | cache_key_value_states                                     |
| --------------------- | ------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| 无缓存self-attention  | [batch_size, seq_len, hd] | None                                                   | None                                                       |
| 有缓存self-attention  | [batch_size, 1, hd]       | None                                                   | ([batch_size, seq_len-1, hd], [batch_size, seq_len-1, hd]) |
| 无缓存cross-attention | [batch_size, seq_len, hd] | ([batch_size, seq_len, hd], [batch_size, seq_len, hd]) | None                                                       |
| 有缓存cross-attention | [batch_size, seq_len, hd] | ([batch_size, 1, hd], [batch_size, 1, hd])             | ([batch_size, seq_len-1, hd], [batch_size, seq_len-1, hd]) |

## 总结

除了缓存，对语料使用sentencepiece或者其他wordpiece方式缩短目标序列长度，也可以加速生成。费劲周折，实现后发现，在k8s的4c机器下，无缓存生成150个token需要30s左右，缓存下为4s左右。但是，如果有块gpu，目标文本长度在200左右的情况下，用不用缓存没多大区别。（ORZ）

## Reference

[1]: https://spaces.ac.cn/archives/8610	"线性Transformer应该不是你要的模型"
[2]: https://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/

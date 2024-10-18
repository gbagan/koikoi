import torch
from safetensors import safe_open

tensors = {}
with safe_open("src-tauri/tensors/discard_sl.safetensors", framework="pt", device="cpu") as f:
    for k in f.keys():
        k2 = k
        k2=k.replace('norm1', 'norm_1')
        k2=k2.replace('norm2', 'norm_2')
        k2=k2.replace("self_attn", "mha")
        k2=k2.replace("linear1", "pwff.linear_inner")
        k2=k2.replace("linear2", "pwff.linear_outer")
        print("in", k)
        print("out", k2, f.get_tensor(k).shape)
        tensors[k2] = f.get_tensor(k)

    in_weight = tensors["encoder_block.attn_encoder.layers.0.mha.in_proj_weight"]
    in_bias = tensors["encoder_block.attn_encoder.layers.0.mha.in_proj_bias"]
    out_weight = tensors["encoder_block.attn_encoder.layers.0.mha.out_proj.weight"]
    out_bias = tensors["encoder_block.attn_encoder.layers.0.mha.out_proj.bias"]
    w_Q, w_K, w_V = in_weight.chunk(3, dim=0)
    b_Q, b_K, b_V = in_bias.chunk(3, dim=0)
    #w_Q = torch.transpose(w_Q, 1, 0)
    #w_K = torch.transpose(w_K, 1, 0)
    #w_V = torch.transpose(w_V, 1, 0)
    #out_weight = torch.transpose(out_weight, 1, 0)

    #m = tensors["encoder_block.attn_encoder.layers.0.pwff.linear_inner.weight"]
    #m = torch.transpose(w_Q, 1, 0)
    #tensors["encoder_block.attn_encoder.layers.0.pwff.linear_inner.weight"] = m

    #m = tensors["encoder_block.attn_encoder.layers.0.pwff.linear_outer.weight"]
    #m = torch.transpose(w_Q, 1, 0)
    #tensors["encoder_block.attn_encoder.layers.0.pwff.linear_outer.weight"] = m

    #m = tensors["encoder_block.attn_encoder.layers.1.pwff.linear_inner.weight"]
    #m = torch.transpose(w_Q, 1, 0)
    #tensors["encoder_block.attn_encoder.layers.1.pwff.linear_inner.weight"] = m

    #m = tensors["encoder_block.attn_encoder.layers.1.pwff.linear_outer.weight"]
    #m = torch.transpose(w_Q, 1, 0)
    #tensors["encoder_block.attn_encoder.layers.1.pwff.linear_outer.weight"] = m

    tensors["encoder_block.attn_encoder.layers.0.mha.query.weight"] = w_Q
    tensors["encoder_block.attn_encoder.layers.0.mha.query.bias"] = b_Q
    tensors["encoder_block.attn_encoder.layers.0.mha.key.weight"] = w_K
    tensors["encoder_block.attn_encoder.layers.0.mha.key.bias"] = b_K
    tensors["encoder_block.attn_encoder.layers.0.mha.value.weight"] = w_V
    tensors["encoder_block.attn_encoder.layers.0.mha.value.bias"] = b_V
    tensors["encoder_block.attn_encoder.layers.0.mha.output.weight"] = out_weight
    tensors["encoder_block.attn_encoder.layers.0.mha.output.bias"] = out_bias
    del tensors["encoder_block.attn_encoder.layers.0.mha.in_proj_weight"]
    del tensors["encoder_block.attn_encoder.layers.0.mha.in_proj_bias"]
    del tensors["encoder_block.attn_encoder.layers.0.mha.out_proj.weight"]
    del tensors["encoder_block.attn_encoder.layers.0.mha.out_proj.bias"]

    in_weight = tensors["encoder_block.attn_encoder.layers.1.mha.in_proj_weight"]
    in_bias = tensors["encoder_block.attn_encoder.layers.1.mha.in_proj_bias"]
    out_weight = tensors["encoder_block.attn_encoder.layers.1.mha.out_proj.weight"]
    out_bias = tensors["encoder_block.attn_encoder.layers.1.mha.out_proj.bias"]
    w_Q, w_K, w_V = in_weight.chunk(3, dim=0)
    b_Q, b_K, b_V = in_bias.chunk(3, dim=0)
    #w_Q = torch.transpose(w_Q, 1, 0)
    #w_K = torch.transpose(w_K, 1, 0)
    #w_V = torch.transpose(w_V, 1, 0)
    #out_weight = torch.transpose(out_weight, 1, 0)
    tensors["encoder_block.attn_encoder.layers.1.mha.query.weight"] = w_Q
    tensors["encoder_block.attn_encoder.layers.1.mha.query.bias"] = b_Q
    tensors["encoder_block.attn_encoder.layers.1.mha.key.weight"] = w_K
    tensors["encoder_block.attn_encoder.layers.1.mha.key.bias"] = b_K
    tensors["encoder_block.attn_encoder.layers.1.mha.value.weight"] = w_V
    tensors["encoder_block.attn_encoder.layers.1.mha.value.bias"] = b_V
    tensors["encoder_block.attn_encoder.layers.1.mha.output.weight"] = out_weight
    tensors["encoder_block.attn_encoder.layers.1.mha.output.bias"] = out_bias
    del tensors["encoder_block.attn_encoder.layers.1.mha.in_proj_weight"]
    del tensors["encoder_block.attn_encoder.layers.1.mha.in_proj_bias"]
    del tensors["encoder_block.attn_encoder.layers.1.mha.out_proj.weight"]
    del tensors["encoder_block.attn_encoder.layers.1.mha.out_proj.bias"]

torch.save(tensors, "discard_sl.pt")
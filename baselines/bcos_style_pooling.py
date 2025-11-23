
import torch
import torch.nn.functional as F
import types

def bcos_style_forward(self, x):
    """
    Forward pass mimicking B-cos AttentionPool2d:
    - Flattens spatial dimensions
    - Adds mean token
    - IGNORES positional embeddings
    - Runs multi-head attention
    """
    # x shape: (N, C, H, W)
    N, C, H, W = x.shape
    
    # Flatten spatial dimensions: (N, C, H*W) -> (H*W, N, C)
    x = x.reshape(N, C, -1).permute(2, 0, 1)
    
    # Add mean token: (H*W + 1, N, C)
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
    
    # CRITICAL DIFFERENCE: NO POSITIONAL EMBEDDING ADDED HERE
    # B-cos implementation skips: x = x + self.positional_embedding[:, None, :]
    
    # Multihead Attention
    # We need to pass the correct weights from the original module
    # Note: The original module has q_proj, k_proj, v_proj, c_proj
    
    # Prepare input projection bias
    in_proj_bias = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])
    
    x, _ = F.multi_head_attention_forward(
        query=x[:1], key=x, value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=self.num_heads,
        q_proj_weight=self.q_proj.weight,
        k_proj_weight=self.k_proj.weight,
        v_proj_weight=self.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0,
        out_proj_weight=self.c_proj.weight,
        out_proj_bias=self.c_proj.bias,
        use_separate_proj_weight=True,
        training=self.training,
        need_weights=False
    )
    
    return x.squeeze(0)

def patch_clip_with_bcos_pooling(model):
    """
    Monkey-patch the CLIP model's attention pooling to behave like B-cos.
    This effectively removes the positional embedding requirement.
    """
    # Bind the new forward method to the attnpool instance
    model.visual.attnpool.forward = types.MethodType(bcos_style_forward, model.visual.attnpool)

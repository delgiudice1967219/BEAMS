import torch
import torch.nn.functional as F

def resize_pos_embed(pos_embed, target_h, target_w):
    """
    Resize positional embeddings to match the target spatial dimensions.
    
    Args:
        pos_embed: (L, C) tensor where L = H_orig * W_orig + 1
        target_h: Target height
        target_w: Target width
        
    Returns:
        Resized positional embeddings (target_h * target_w + 1, C)
    """
    # Assume standard CLIP ResNet50 with 7x7 spatial grid
    # pos_embed shape is (50, 2048) for RN50
    
    class_pos_embed = pos_embed[0:1]  # (1, C)
    spatial_pos_embed = pos_embed[1:] # (HW, C)
    
    # Determine original spatial size (assuming square)
    hw = spatial_pos_embed.shape[0]
    h_orig = w_orig = int(hw ** 0.5)
    
    if h_orig * w_orig != hw:
        # Fallback or error if not square, but standard CLIP is 7x7
        raise ValueError(f"Positional embedding size {hw} does not match a square grid.")
        
    # Reshape to (1, C, H, W) for interpolation
    spatial_pos_embed = spatial_pos_embed.reshape(1, h_orig, w_orig, -1).permute(0, 3, 1, 2)
    
    # Interpolate
    spatial_pos_embed = F.interpolate(
        spatial_pos_embed, 
        size=(target_h, target_w), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Reshape back to (HW, C)
    spatial_pos_embed = spatial_pos_embed.permute(0, 2, 3, 1).reshape(target_h * target_w, -1)
    
    return torch.cat([class_pos_embed, spatial_pos_embed], dim=0)


def attention_pool_forward_patch(self, x):
    """
    Patched forward method for AttentionPool2d to handle arbitrary input sizes.
    """
    # x shape: (N, C, H, W)
    N, C, H, W = x.shape
    
    # Flatten spatial dimensions: (N, C, H*W) -> (H*W, N, C)
    x = x.reshape(N, C, -1).permute(2, 0, 1)
    
    # Add mean token: (H*W + 1, N, C)
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
    
    # Get positional embedding
    pos_embed = self.positional_embedding.to(x.dtype)
    
    # Resize if needed
    if pos_embed.shape[0] != x.shape[0]:
        pos_embed = resize_pos_embed(pos_embed, H, W)
        
    # Add positional embedding: (H*W + 1, 1, C) broadcast to (H*W + 1, N, C)
    x = x + pos_embed[:, None, :].to(x.device)
    
    # Multihead Attention
    x, _ = F.multi_head_attention_forward(
        query=x[:1], key=x, value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=self.num_heads,
        q_proj_weight=self.q_proj.weight,
        k_proj_weight=self.k_proj.weight,
        v_proj_weight=self.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
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


def patch_clip_attnpool(model):
    """
    Apply the patch to the CLIP model's visual attention pooling layer.
    """
    import types
    # Bind the new method to the instance
    model.visual.attnpool.forward = types.MethodType(attention_pool_forward_patch, model.visual.attnpool)

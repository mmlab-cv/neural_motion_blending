import torch

def slerp(v0, v1, alpha, dot_threshold=0.9995):
    """
    v0, v1: [143, 13, 200] (Joints, Features, Time)
    alpha: float or [200] tensor (Interpolation weight)
    """
    # 1. Flatten the spatial dimensions: [143*13, 200] -> [1859, 200]
    j, f, t = v0.shape
    v0_flat = v0.view(-1, t)
    v1_flat = v1.view(-1, t)

    # 2. Compute the norm (magnitude) of the noise at each frame
    # DDIM noise should have a norm roughly equal to sqrt(Dim)
    v0_norm = torch.linalg.norm(v0_flat, dim=0, keepdim=True) # [1, 200]
    v1_norm = torch.linalg.norm(v1_flat, dim=0, keepdim=True) # [1, 200]

    # 3. Compute the dot product across the 1859 features per frame
    dot = torch.sum(v0_flat * v1_flat, dim=0, keepdim=True)
    
    # 4. Normalize the dot product for the cosine calculation
    # This is critical to prevent NaNs in acos
    dot_norm = dot / (v0_norm * v1_norm + 1e-9)
    dot_norm = dot_norm.clamp(-1.0, 1.0)

    # 5. Handle Alpha broadcasting
    # If alpha is a scalar, it works. If it's a [200] tensor, we need [1, 200]
    if isinstance(alpha, torch.Tensor) and alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)

    # 6. Slerp Logic
    theta_0 = torch.acos(dot_norm)
    theta_t = theta_0 * alpha
    
    sin_theta_0 = torch.sin(theta_0)
    
    # Selection mask for nearly parallel vectors (to avoid div by zero)
    is_lerp = dot_norm > dot_threshold
    
    # Standard Slerp coefficients
    s0 = torch.sin(theta_0 - theta_t) / (sin_theta_0 + 1e-9)
    s1 = torch.sin(theta_t) / (sin_theta_0 + 1e-9)
    
    res_slerp = s0 * v0_flat + s1 * v1_flat
    res_lerp = (1.0 - alpha) * v0_flat + alpha * v1_flat
    
    # Choose between Slerp and Lerp
    res = torch.where(is_lerp, res_lerp, res_slerp)

    # 7. Restore back to original shape [143, 13, 200]
    return res.view(j, f, t)
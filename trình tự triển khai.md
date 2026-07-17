Giai đoạn 1: Data pipeline
    PIL image
    → transform
    → tensor
    → batch
    → kiểm tra min/max/mean/std/shape

Giai đoạn 2: Image encoder
    kiểm tra shape qua từng layer
    chưa cần decoder

Giai đoạn 3: Gaussian head
    feature vector
    → mu, log_var

Giai đoạn 4: Reparameterization
    mu, log_var
    → std, epsilon, z
    → kiểm tra gradient

Giai đoạn 5: Image decoder
    z
    → reconstructed image đúng kích thước

Giai đoạn 6: VanillaVAE hoàn chỉnh
    encode
    reparameterize
    decode
    forward

Giai đoạn 7: Loss
    reconstruction
    KL
    total loss

Giai đoạn 8: Training loop
    forward
    backward
    optimizer step
    validation
    checkpoint

Giai đoạn 9: Test và phân tích
    reconstruction
    random sampling
    latent interpolation
    latent distribution
    ablation
vanilla_vae_research/
│
├── configs/
│   ├── image_64.yaml
│   ├── image_128.yaml
│   └── image_256.yaml
│
├── src/
│   ├── data/
│   │   ├── image_dataset.py
│   │   ├── transforms.py
│   │   ├── dataloader.py
│   │   └── batch.py
│   │
│   ├── models/
│   │   ├── base_vae.py
│   │   ├── outputs.py
│   │   │
│   │   ├── encoders/
│   │   │   └── image_encoder.py
│   │   │
│   │   ├── latent/
│   │   │   ├── gaussian_head.py
│   │   │   └── reparameterization.py
│   │   │
│   │   ├── decoders/
│   │   │   └── image_decoder.py
│   │   │
│   │   └── vanilla_vae.py
│   │
│   ├── losses/
│   │   ├── reconstruction.py
│   │   ├── kl_divergence.py
│   │   └── vae_loss.py
│   │
│   ├── engine/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   │
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       ├── checkpoint.py
│       ├── shape_trace.py
│       └── visualization.py
│
├── tests/
│   ├── test_image_dataset.py
│   ├── test_encoder.py
│   ├── test_reparameterization.py
│   ├── test_decoder.py
│   ├── test_loss.py
│   └── test_vanilla_vae.py
│
├── inspect_data.py
├── inspect_model.py
├── train.py
├── test.py
├── reconstruct.py
├── sample.py
└── requirements.txt
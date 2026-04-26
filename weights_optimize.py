import optuna
import copy
import torch
import yaml
from data import MNISTDataset, data_split
from model import VAE
from utils import vae_loss_fn_ver3, vae_loss_fn_ver1, vae_loss_fn_ver2, run_training, run_training_optim
from optuna.samplers import TPESampler


LOSS_REGISTRY = {
    "loss_fn_1": {
        "fn": vae_loss_fn_ver1,
        "search_space": {
            "lambda_rec": (0.3, 2.0, False),
            "lambda_kl": (1e-4, 1e-2, True),
        },
        "fixed_params": {
            "lambda_ssim": 0.0
        }
    },
    "loss_fn_2": {
        "fn": vae_loss_fn_ver2,
        "search_space": {
            "lambda_rec": (0.3, 2.0, False),
            "lambda_kl": (1e-4, 1e-2, True),
            "lambda_ssim": (0.3, 2.0, False),
        },
        "fixed_params": {}
    },
    "loss_fn_3": {
        "fn": vae_loss_fn_ver3,
        "search_space": {
            "lambda_rec": (0.3, 2.0, False),
            "lambda_kl": (1e-4, 1e-2, True),
            "lambda_ssim": (0.3, 2.0, False),
        },
        "fixed_params": {}
    }
}

def objective(trial, base_config, loss_name, loss_info, train_loader, val_loader, device):
    trial_config = copy.deepcopy(base_config)

    trial_config["num_epochs"] = 3
    trial_config["patience"] = min(base_config.get("patience", 2), 2)

    for key, value in loss_info["fixed_params"].items():
        trial_config[key] = value

    for param_name, (low, high, use_log) in loss_info["search_space"].items():
        trial_config[param_name] = trial.suggest_float(param_name, low, high, log=use_log)

    model_vae_trial = VAE(latent_features=trial_config["latent_features"])

    _, history = run_training_optim(
        model=model_vae_trial,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trial_config,
        device=device,
        loss_fn=loss_info["fn"]
    )

    if len(history["val_loss"]) == 0:
        return float("inf")

    best_val_loss = min(history["val_loss"])

    if not torch.isfinite(torch.tensor(best_val_loss)):
        return float("inf")

    return best_val_loss

# Đọc dữ liệu config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chia dữ liệu
full_train_dataset = MNISTDataset(train=True, download=False)
train_loader, val_loader = data_split(full_train_dataset=full_train_dataset, config=config)

print("\n")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device: ", device)


print("Bắt đầu quá trình Bayesian Optimization với Optuna...")

dataset_name = "MNIST"
results = {}

for loss_name, loss_info in LOSS_REGISTRY.items():
    print(f"\nBat dau toi uu cho {loss_name} ...")

    sampler = TPESampler(n_startup_trials=5, seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda trial: objective(
            trial=trial,
            base_config=config,
            loss_name=loss_name,
            loss_info=loss_info,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        ),
        n_trials=10
    )

    results[loss_name] = {
        "best_value": study.best_value,
        "best_params": study.best_params
    }

    config[dataset_name][loss_name].update(study.best_params)

    if "lambda_ssim" not in study.best_params:
        config[dataset_name][loss_name]["lambda_ssim"] = 0.0

    print(f"{loss_name} - best val loss: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

with open("config_optimized.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

print("\nDa luu bo trong so toi uu vao config_optimized.yaml")

# # TPESampler sử dụng Expected Improvement (EI) làm acquisition function
# bo_sampler = TPESampler(
#     n_startup_trials = 5,
#     seed=42
# )
# study = optuna.create_study(direction="minimize", sampler=bo_sampler)

# # Chạy tối ưu hóa trong n_trials vòng (ví dụ: 20 vòng thử nghiệm). 
# study.optimize(lambda trial: objective(trial, config), n_trials=20)

# print("\nQuá trình tìm kiếm đã hoàn tất!")

# # --- IN RA KẾT QUẢ TỐT NHẤT ---
# best_trial = study.best_trial
# print(f"Giá trị Validation Loss cho Loss Function 1 tốt nhất đạt được: {best_trial.value:.4f}")
# print("Bộ tham số hoàn hảo nhất là:")
# for key, value in best_trial.params.items():
#     print(f"    {key}: {value}")

# # Xem quá trình giảm loss qua từng trial
# fig_optimization = optuna.visualization.plot_optimization_history(study)
# fig_optimization.show()

# # Đánh giá xem giữa lambda_rec, lambda_kl, lambda_ssim, cái nào quan trọng nhất
# fig_importance = optuna.visualization.plot_param_importances(study)
# fig_importance.show()

# # Xem sự phân bố của các cấu hình đã thử (giúp biết vùng giá trị nào là tối ưu)
# fig_slice = optuna.visualization.plot_slice(study)
# fig_slice.show()
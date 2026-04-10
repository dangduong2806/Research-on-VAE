import optuna
import copy
import torch
import yaml
from data import MNISTDataset, data_split, CIFARDataset
from model import VAE
from cifar_model import CIFAR_VAE
from utils import vae_loss_fn_ver3, vae_loss_fn_ver1, vae_loss_fn_ver2, run_training, run_training_optim
from optuna.samplers import TPESampler

def objective(trial, config):
    suggested_lambda_rec = trial.suggest_float('lambda_rec', 0.3, 2.0)
    # KL loss thường nhỏ => Tìm trong không gian logarit
    suggested_lambda_kl = trial.suggest_float('lambda_kl', 1e-4, 1e-2, log=True)

    suggested_lambda_ssim = trial.suggest_float('lambda_ssim', 0.3, 2.0)

    # Tạo bản sao của config và cập nhật các trọng số mới
    trial_config = copy.deepcopy(config)
    trial_config['lambda_rec'] = suggested_lambda_rec
    trial_config['lambda_kl'] = suggested_lambda_kl
    trial_config['lambda_ssim'] = suggested_lambda_ssim

    trial_config['num_epochs'] = 5 # 3-5. Sau khi tìm được bộ lambda tốt => train full

    # model_vae_trial = VAE(latent_features=trial_config['latent_features'])
    model_vae_trial = CIFAR_VAE(latent_features=trial_config['latent_features'], in_channels=trial_config['in_channels'])

    # Huấn luyện
    _, history = run_training_optim(
        model=model_vae_trial,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trial_config,
        device=device,
        loss_fn=vae_loss_fn_ver3
    )

    # Lấy giá trị validation loss của epoch cuối cùng làm thước đo
    # final_val_loss = history['val_loss'][-1]
    # Lấy best val loss trong quá trình train
    final_val_loss = min(history['val_loss'])

    # Nếu mô hình bị phân kỳ (loss = NaN), trả về vô cùng lớn để Optuna loại bỏ
    if torch.isnan(torch.tensor(final_val_loss)):
        return float('inf')
    
    return final_val_loss

# Đọc dữ liệu config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chia dữ liệu
# full_train_dataset = MNISTDataset(train=True, download=False)
full_train_dataset = CIFARDataset(train=True, download=False)
train_loader, val_loader = data_split(full_train_dataset=full_train_dataset, config=config)

print("\n")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device: ", device)


print("Bắt đầu quá trình Bayesian Optimization với Optuna...")

# TPESampler sử dụng Expected Improvement (EI) làm acquisition function
bo_sampler = TPESampler(
    n_startup_trials = 5,
    seed=42
)
study = optuna.create_study(direction="minimize", sampler=bo_sampler)

# Chạy tối ưu hóa trong n_trials vòng (ví dụ: 20 vòng thử nghiệm). 
study.optimize(lambda trial: objective(trial, config), n_trials=20)

print("\nQuá trình tìm kiếm đã hoàn tất!")

# --- IN RA KẾT QUẢ TỐT NHẤT ---
best_trial = study.best_trial
print(f"Giá trị Validation Loss cho Loss Function 3 tốt nhất đạt được: {best_trial.value:.4f}")
print("Bộ tham số hoàn hảo nhất là:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Xem quá trình giảm loss qua từng trial
fig_optimization = optuna.visualization.plot_optimization_history(study)
fig_optimization.show()

# Đánh giá xem giữa lambda_rec, lambda_kl, lambda_ssim, cái nào quan trọng nhất
fig_importance = optuna.visualization.plot_param_importances(study)
fig_importance.show()

# Xem sự phân bố của các cấu hình đã thử (giúp biết vùng giá trị nào là tối ưu)
fig_slice = optuna.visualization.plot_slice(study)
fig_slice.show()
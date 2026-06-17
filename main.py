import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.models.quantile_regression import QuantileRegressionV1

# Thư mục gốc của project (nơi chứa main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


models_info = {
    "Adam": {
        "best_theta": os.path.join(BASE_DIR, "results", "weights", "adam", "adam_theta01.npy"),
        "losses": {
            "lr = 0.1":  os.path.join(BASE_DIR, "results", "weights", "adam", "adam_loss1.npy"),
            "lr = 0.05": os.path.join(BASE_DIR, "results", "weights", "adam", "adam_loss05.npy"),
            "lr = 0.01": os.path.join(BASE_DIR, "results", "weights", "adam", "adam_loss01.npy")
        }
    },
    "Momentum": {
        "best_theta": os.path.join(BASE_DIR, "results", "weights", "momentum", "momentum_weights_005.npy"),
        "losses": {
            "lr = 0.1":  os.path.join(BASE_DIR, "results", "weights", "momentum", "momentum_loss_history_01.npy"),
            "lr = 0.05": os.path.join(BASE_DIR, "results", "weights", "momentum", "momentum_loss_history_005.npy"),
            "lr = 0.01": os.path.join(BASE_DIR, "results", "weights", "momentum", "momentum_loss_history_001.npy")
        }
    },
    "Nesterov": {
        "best_theta": os.path.join(BASE_DIR, "results", "weights", "Nesterov", "nesterov_weights.npy"),
        "losses": {
            "lr = 0.1":  os.path.join(BASE_DIR, "results", "weights", "Nesterov", "nesterov_loss_1.npy"),
            "lr = 0.05": os.path.join(BASE_DIR, "results", "weights", "Nesterov", "nesterov_loss_05.npy"),
            "lr = 0.01": os.path.join(BASE_DIR, "results", "weights", "Nesterov", "nesterov_loss_01.npy")
        }
    },
    "RMSProp": {
        "best_theta": os.path.join(BASE_DIR, "results", "weights", "RMSProp", "rms_weights.npy"),
        "losses": {
            "lr = 0.1":  os.path.join(BASE_DIR, "results", "weights", "RMSProp", "rms_loss_1.npy"),
            "lr = 0.05": os.path.join(BASE_DIR, "results", "weights", "RMSProp", "rms_loss_05.npy"),
            "lr = 0.01": os.path.join(BASE_DIR, "results", "weights", "RMSProp", "rms_loss_01.npy")
        }
    }
}

lr_colors = {
    "lr = 0.1":  "#e74c3c", 
    "lr = 0.05": "#3498db", 
    "lr = 0.01": "#2ecc71" 
}

def main_evaluate_and_plot(X_test, y_test):
    print(" BẮT ĐẦU ĐÁNH GIÁ 4 THUẬT TOÁN (SỬ DỤNG THETA TỐT NHẤT)...")
    
    results = []
    model = QuantileRegressionV1(tau=0.9)

    #ĐÁNH GIÁ ĐIỂM SỐ TRÊN TẬP TEST
    for name, info in models_info.items():
        try:
            model.load_weights(info["best_theta"])
            
            y_pred = model.predict(X_test)
            p_loss = model.pinball_loss(y_test, y_pred)
            mae_val = mae(y_test, y_pred)
            rmse_val = rmse(y_test, y_pred)
            
            results.append({
                "Thuật Toán": name,
                "Pinball Loss (tau=0.9)": np.round(p_loss, 4),
                "MAE": np.round(mae_val, 4),
                "RMSE": np.round(rmse_val, 4)
            })
            print(f" Đã chấm điểm xong thuật toán: {name}")
            
        except FileNotFoundError as e:
            print(f" Lỗi: Không tìm thấy file theta của {name}")
            print(e)

    df_results = pd.DataFrame(results).set_index("Thuật Toán")
    df_results = df_results.sort_values(by="Pinball Loss (tau=0.9)") 
    print("\n THUẬT TOÁN TỐT NHẤT:")
    print("-" * 65)
    print(df_results)
    print("-" * 65)

    print("\n Đang vẽ 4 biểu đồ phân tích Learning Rate...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten() 
    
    for i, (name, info) in enumerate(models_info.items()):
        ax = axes[i]
        
        #
        for lr_label, loss_path in info["losses"].items():
            try:
                loss_data = np.load(loss_path)
                ax.plot(loss_data, label=lr_label, color=lr_colors[lr_label], linewidth=2, alpha=0.8)
            except FileNotFoundError:
                print(f" Thiếu file loss {lr_label} của {name}")
        
        ax.set_title(f"{name} Optimizer", fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel("Vòng lặp (Epochs)", fontsize=11)
        ax.set_ylabel("Pinball Loss", fontsize=11)
        ax.legend(title="Learning Rate", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle('Tác Động Của Learning Rate Đến Quá Trình Hội Tụ', fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    plots_dir = os.path.join(BASE_DIR, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    final_plot_path = os.path.join(plots_dir, "final_lr_comparison_subplots.png")
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    

test_path = os.path.join(BASE_DIR, "dataset", "foodDeli_processed", "test_processed.csv")
df_test = pd.read_csv(test_path).dropna()
target_column = 'Time_taken' 
columns_to_drop = ['ID', target_column]

X_test_raw = df_test.drop(columns=columns_to_drop, errors='ignore').values
y_test = df_test[target_column].values

def add_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

X_test_final = add_bias(X_test_raw)

# Kích hoạt chạy toàn bộ luồng
main_evaluate_and_plot(X_test_final, y_test)

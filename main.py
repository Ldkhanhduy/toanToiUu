import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import QuantileRegressor
from src.models.quantile_regression import QuantileRegressionV1

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

models_info = {
    "Adam": {
        "best_theta": "results/weights/adam/adam_theta1.npy",
        "best_loss_path": "results/weights/adam/adam_loss1.npy",
        "color": "#e74c3c"
    },
    "Momentum": {
        "best_theta": "results/weights/momentum/momentum_weights_005.npy",
        "best_loss_path": "results/weights/momentum/momentum_loss_history_005.npy",
        "color": "#3498db"
    },
    "Nesterov": {
        "best_theta": "results/weights/Nesterov/nesterov_weights.npy",
        "best_loss_path": "results/weights/Nesterov/nesterov_loss_1.npy",
        "color": "#2ecc71"
    },
    "RMSProp": {
        "best_theta": "results/weights/RMSProp/rms_weights.npy",
        "best_loss_path": "results/weights/RMSProp/rms_loss_01.npy",
        "color": "#9b59b6"
    }
}

def main_evaluate_and_plot(X_test_final, X_test_raw, y_test, X_train_raw, y_train):
    
    results = []
    model_handmade = QuantileRegressionV1(tau=0.9)

    # ĐÁNH GIÁ 4 THUẬT TOÁN 
    for name, info in models_info.items():
        try:
            model_handmade.load_weights(info["best_theta"])
            y_pred = model_handmade.predict(X_test_final)
            
            p_loss = model_handmade.pinball_loss(y_test, y_pred)
            mae_val = mae(y_test, y_pred)
            rmse_val = rmse(y_test, y_pred)
            
            results.append({
                "Thuật Toán": f"{name} (Hand-made)", 
                "Pinball Loss (tau=0.9)": np.round(p_loss, 4),
                "MAE": np.round(mae_val, 4),
                "RMSE": np.round(rmse_val, 4)
            })
        except FileNotFoundError:
            print(f" Không tìm thấy file trọng số tốt nhất của {name}")

    # ĐÁNH GIÁ THEO SKLEARN
    print("  Đang chạy mô hình hồi quy phân vị của Sklearn (Huấn luyện trên tập Train)...")
    try:
        sklearn_model = QuantileRegressor(quantile=0.9, alpha=0.0, solver='highs')
        sklearn_model.fit(X_train_raw, y_train)
        y_pred_sk = sklearn_model.predict(X_test_raw)
        # Chấm điểm
        p_loss_sk = model_handmade.pinball_loss(y_test, y_pred_sk)
        mae_sk = mae(y_test, y_pred_sk)
        rmse_sk = rmse(y_test, y_pred_sk)
        
        results.append({
            "Thuật Toán": "Sklearn (QuantileRegressor)", 
            "Pinball Loss (tau=0.9)": np.round(p_loss_sk, 4),
            "MAE": np.round(mae_sk, 4),
            "RMSE": np.round(rmse_sk, 4)
        })
    except Exception as e:
        print(f"  Lỗi khi chạy Sklearn: {e}")

    # BẢNG XẾP HẠNG
    df_results = pd.DataFrame(results).set_index("Thuật Toán")
    df_results = df_results.sort_values(by="Pinball Loss (tau=0.9)")
    print("\n BẢNG XẾP HẠNG:")
    
    print(df_results)
    

    print("\n Đang vẽ biểu đồ so sánh tiến trình hội tụ tối ưu...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    for name, info in models_info.items():
        try:
            loss_data = np.load(info["best_loss_path"])
            plt.plot(loss_data, label=f"{name}", color=info["color"], linewidth=2.5, alpha=0.9)
        except FileNotFoundError:
            print(f" Thiếu dữ liệu file tiến trình loss tốt nhất của {name}")

    plt.title('So Sánh Tiến Trình Hội Tụ Của Các Thuật Toán Tối Ưu', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Pinball Loss', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    combined_plot_path = os.path.join(plots_dir, "best_optimizers_comparison.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_path = "dataset/foodDeli_processed/test_processed.csv"
    df_test = pd.read_csv(test_path).dropna()

    train_path = "dataset/foodDeli_processed/train_processed.csv" 
    df_train = pd.read_csv(train_path).dropna()
    
    target_column = 'Time_taken' 
    columns_to_drop = ['ID', target_column]

    X_test_raw = df_test.drop(columns=columns_to_drop, errors='ignore').values
    y_test = df_test[target_column].values

    X_train_raw = df_train.drop(columns=columns_to_drop, errors='ignore').values
    y_train = df_train[target_column].values

    def add_bias(X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    X_test_final = add_bias(X_test_raw)
    
    main_evaluate_and_plot(X_test_final, X_test_raw, y_test, X_train_raw, y_train)
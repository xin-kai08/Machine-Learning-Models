import os
from run_experiments_f4 import (
    train_and_search_model,
    LSTMClassifier,
    ChargeSequenceDataset3D
)

# === Sweep 參數 ===
batch_sizes = [8, 16, 32]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
seq_lens = [10, 20, 30, 40]
strides = [1]
num_epochs = 100

# === 複雜度 sweep 參數 ===
num_layers_list = [1, 2, 3, 4]
hidden_dim_list = [16, 32, 64]
dropout_rates = [0.0, 0.3]

# === 結果總資料夾 ===
BASE_DATA_DIR = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware\preprocessed_kfold"
BASE_RESULT_DIR = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\complexity compare"

# 確保總資料夾存在
os.makedirs(BASE_RESULT_DIR, exist_ok=True)

# === 開始跑不同組合 ===
for stride in strides:
    base_preprocessed_dir = os.path.join(BASE_DATA_DIR, f"stride_{stride}")
    
    for num_layers in num_layers_list:
        for hidden_dim in hidden_dim_list:
            for dropout_rate in dropout_rates:
                lstm_args = {
                    'input_dim': 4,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_classes': 4,
                    'dropout_rate': dropout_rate
                }

                result_dir = os.path.join(
                    BASE_RESULT_DIR,
                    f"stride_{stride}",
                    f"layers_{num_layers}_hidden_{hidden_dim}_dropout_{dropout_rate}"
                )

                train_and_search_model(
                    LSTMClassifier,
                    lstm_args,
                    ChargeSequenceDataset3D,
                    "X_seq{seq_len}_3d.npy",
                    "y_seq{seq_len}_3d.npy",
                    batch_sizes,
                    learning_rates,
                    seq_lens,
                    RESULT_DIR=result_dir,
                    num_epochs=num_epochs,
                    base_preprocessed_dir=base_preprocessed_dir
                )
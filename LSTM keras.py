import os
import json
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =====================
# 路徑與資料配置（與原檔案保持一致）
# =====================
BASE_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware"
RESULT_DIR = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\result\keras\20250819"

LABEL_DIRS = {
    0: os.path.join(BASE_PATH, "normal"),
    1: os.path.join(BASE_PATH, "abnormal/wire_rust"),
    2: os.path.join(BASE_PATH, "abnormal/transformer_rust"),
    3: os.path.join(BASE_PATH, "abnormal/transformer_overheating"),
}

# =====================
# 超參數（可被實驗迴圈覆寫）
# =====================
MAX_SEQ_LEN = 15
STRIDE = 5

INPUT_DIM = 4
HIDDEN_DIM = 16
NUM_LAYERS = 1
NUM_CLASSES = len(LABEL_DIRS)
NUM_EPOCHS = 100

KFOLD_SPLITS = 5
SEED = 42

# 匯出控制
EXPORT_TFLITE = True          # 匯出 .tflite
SAVE_SCALER_JSON = False      # 每個 fold 另存 scaler 參數（mean/std）
CLEAR_RESULT_DIR_BEFORE_RUN = True

os.makedirs(RESULT_DIR, exist_ok=True)
np.random.seed(SEED)

# =====================
# 資料處理
# =====================

def process_file(file_path, label, sequences, labels, max_seq_len=MAX_SEQ_LEN):
    """
    讀取 CSV 檔案並切分為長度為 max_seq_len 的小段（滑動視窗）。
    假設 CSV 欄位：'current', 'voltage', 'power', 'temp_C'
    """
    df = pd.read_csv(file_path)
    current = df['current'].values
    voltage = df['voltage'].values
    power = df['power'].values
    temp_C = df['temp_C'].values

    # 與原稿一致：欄位順序 [current, voltage, temp_C, power]
    sequence = np.column_stack((current, voltage, power, temp_C)).astype(np.float32)
    seq_len = sequence.shape[0]
    for start in range(0, seq_len - max_seq_len + 1, STRIDE):
        end = start + max_seq_len
        chunk = sequence[start:end]
        sequences.append(chunk)
        labels.append(label)


def load_data():
    sequences, labels = [], []
    for label, folder in LABEL_DIRS.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(folder, filename)
                process_file(file_path, label, sequences, labels, max_seq_len=MAX_SEQ_LEN)
    sequences = np.asarray(sequences, dtype=np.float32)  # (N, T, F)
    labels = np.asarray(labels, dtype=np.int64)

    print("sequences shape:", sequences.shape)
    print("labels shape:", labels.shape)
    for i in range(NUM_CLASSES):
        print(f"Number of class {i} samples:", np.sum(labels == i))
    print("Unique labels:", np.unique(labels))
    return sequences, labels


# =====================
# 模型定義（Keras 版）
# =====================

def build_lstm_model(input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float= 0.0):
    inputs = keras.Input(shape=(MAX_SEQ_LEN, input_dim))
    x = inputs
    # 疊層 LSTM：前 n-1 層 return_sequences=True，最後一層 False
    for i in range(num_layers - 1):
        x = layers.LSTM(hidden_dim, return_sequences=True)(x)
    x = layers.LSTM(hidden_dim, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes)(x)  # logits

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')]
    )
    return model


# =====================
# 評估輔助
# =====================

def evaluate_np(model: keras.Model, x: np.ndarray, y: np.ndarray):
    """回傳 loss, acc, precision, recall, f1（macro）。"""
    logits = model.predict(x, verbose=0)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = float(loss_fn(y, logits).numpy())
    preds = logits.argmax(axis=1)
    acc = float((preds == y).mean()) if len(y) > 0 else 0.0
    precision = precision_score(y, preds, average='macro', zero_division=0)
    recall = recall_score(y, preds, average='macro', zero_division=0)
    f1 = f1_score(y, preds, average='macro', zero_division=0)
    return loss, acc, precision, recall, f1, preds


# =====================
# K-fold 訓練
# =====================

def kfold_training(sequences: np.ndarray, labels: np.ndarray):
    kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=SEED)

    fold_final_metrics = []
    all_folds_metrics = []  # list of DataFrame（每個 fold 全部 epoch）

    fold_idx = 0
    for train_idx, test_idx in kfold.split(sequences, labels):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx} / {KFOLD_SPLITS} ===")

        x_train_raw, y_train = sequences[train_idx], labels[train_idx]
        x_test_raw, y_test = sequences[test_idx], labels[test_idx]

        # StandardScaler：fit on train (合併時間維度)；transform train/test
        scaler = StandardScaler()
        T, F = x_train_raw.shape[1], x_train_raw.shape[2]
        x_train_flat = x_train_raw.reshape(-1, F)
        scaler.fit(x_train_flat)
        x_train = scaler.transform(x_train_flat).reshape(x_train_raw.shape)

        x_test_flat = x_test_raw.reshape(-1, F)
        x_test = scaler.transform(x_test_flat).reshape(x_test_raw.shape)

        if SAVE_SCALER_JSON:
            scaler_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_scaler.json")
            scaler_dict = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
                'var': scaler.var_.tolist(),
                'n_features_in_': int(scaler.n_features_in_),
            }
            with open(scaler_path, 'w', encoding='utf-8') as f:
                json.dump(scaler_dict, f, ensure_ascii=False, indent=2)
            print(f"Scaler params saved to {scaler_path}")

        # 建立模型
        model = build_lstm_model(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, dropout= 0.0)

        # Pre-train（隨機初始）表現
        pre_loss, pre_acc, _, _, _, _ = evaluate_np(model, x_test, y_test)
        print(f"Pre-training  | Test Loss: {pre_loss:.4f}, Test Acc: {pre_acc:.4f}")
        print("===========================================")

        # 記錄每 epoch 指標
        history_rows = []

        # 訓練
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            h = model.fit(
                x_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=0,
                shuffle=True,
                validation_data=(x_test, y_test)
            )
            # 訓練/測試指標
            train_loss = float(h.history['loss'][-1])
            train_acc = float(h.history['acc'][-1])
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_np(model, x_test, y_test)

            history_rows.append({
                'Epoch': epoch + 1,
                'Train Loss': train_loss,
                'Train Accuracy': train_acc,
                'Test Loss': test_loss,
                'Test Accuracy': test_acc,
                'Test Precision': test_prec,
                'Test Recall': test_rec,
                'Test F1-score': test_f1,
                'Fold': fold_idx,
            })

            print(
                f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
                f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}"
            )

        elapsed = time.time() - start_time
        print(f"⌛訓練時間：{elapsed:.2f} 秒")

        # Post-train 指標（以最後一個 epoch 的模型）
        post_loss, post_acc, post_prec, post_rec, post_f1, preds = evaluate_np(model, x_test, y_test)
        print(f"Post-training | Test Loss: {post_loss:.4f}, Test Acc: {post_acc:.4f}")

        fold_final_metrics.append({
            'Fold': fold_idx,
            'Pre-train Loss': pre_loss,
            'Pre-train Accuracy': pre_acc,
            'Post-train Loss': post_loss,
            'Post-train Accuracy': post_acc,
            'Precision': post_prec,
            'Recall': post_rec,
            'F1-Score': post_f1,
        })

        # 混淆矩陣
        cm = confusion_matrix(y_test, preds, labels=list(range(NUM_CLASSES)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(NUM_CLASSES)))
        plt.figure(figsize=(4, 4))
        disp.plot(values_format='d', cmap='Blues')
        plt.title(f"Fold {fold_idx} - Confusion Matrix")
        cm_pdf_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_cm.pdf")
        cm_svg_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_cm.svg")
        plt.savefig(cm_pdf_path, bbox_inches='tight')
        plt.savefig(cm_svg_path, bbox_inches='tight')
        plt.close()

        # 儲存模型（.keras）
        model_save_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_model.keras")
        model.save(model_save_path)
        print(f"Model for fold {fold_idx} saved to {model_save_path}")

        # 匯出 TFLite（可選）
        if EXPORT_TFLITE:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,       # 內建 TFLite ops
                tf.lite.OpsSet.SELECT_TF_OPS          # 允許少量 TF ops（例如 TensorList 相關）
            ]
            converter._experimental_lower_tensor_list_ops = False
            # 先給一個 baseline（FP32）
            tflite_model = converter.convert()
            tflite_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_model_fp32.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite (FP32) for fold {fold_idx} saved to {tflite_path}")

            # 也可附帶動態範圍量化版本
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_dyn = converter.convert()
            tflite_dyn_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_model_dynamic.tflite")
            with open(tflite_dyn_path, 'wb') as f:
                f.write(tflite_dyn)
            print(f"TFLite (Dynamic Quant) for fold {fold_idx} saved to {tflite_dyn_path}")

        # 整理 epoch 指標 DataFrame
        fold_metrics_df = pd.DataFrame(history_rows)
        all_folds_metrics.append(fold_metrics_df)

        # 對齊原版：輸出每個 fold 的曲線圖
        plot_metric_curves([fold_metrics_df])

    # 摘要
    final_metrics_df = pd.DataFrame(fold_final_metrics)
    print("\n=== Final Metrics Summary Across All Folds ===")
    print(final_metrics_df.to_string(index=False))
    return final_metrics_df, all_folds_metrics


# =====================
# 視覺化（與原版相容）
# =====================

def plot_metric_curves(all_folds_metrics):
    for fold_df in all_folds_metrics:
        fold = int(fold_df['Fold'].iloc[0])
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Fold {fold} Metrics Curves', fontsize=16)
        epochs = fold_df['Epoch']

        axes[0, 0].plot(epochs, fold_df['Train Loss'], label='Train Loss')
        axes[0, 0].set_title('Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, fold_df['Test Loss'], label='Test Loss')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()

        axes[0, 2].plot(epochs, fold_df['Train Accuracy'], label='Train Accuracy')
        axes[0, 2].set_title('Train Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()

        axes[1, 0].plot(epochs, fold_df['Test Accuracy'], label='Test Accuracy')
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, fold_df['Test Precision'], label='Test Precision')
        axes[1, 1].set_title('Test Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()

        axes[1, 2].plot(epochs, fold_df['Test F1-score'], label='Test F1-score')
        axes[1, 2].set_title('Test F1-score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-score')
        axes[1, 2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_pdf = os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.pdf")
        plot_svg = os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.svg")
        plt.savefig(plot_pdf, bbox_inches='tight')
        plt.savefig(plot_svg, bbox_inches='tight')
        plt.close(fig)


def plot_overlaid_metrics(all_folds_metrics):
    # 假設所有 fold 的 epoch 數相同
    if not all_folds_metrics:
        return
    epochs = all_folds_metrics[0]['Epoch']

    metrics = {
        'Train Accuracy': 'Train Accuracy',
        'Test Accuracy': 'Test Accuracy',
        'Train Loss': 'Train Loss',
        'Test Loss': 'Test Loss',
        'Test Precision': 'Test Precision',
        'Test Recall': 'Test Recall',
        'Test F1-score': 'Test F1-score'
    }

    for metric_title, col in metrics.items():
        plt.figure(figsize=(10, 6))
        for fold_df in all_folds_metrics:
            fold = int(fold_df['Fold'].iloc[0])
            plt.plot(epochs, fold_df[col], label=f'Fold {fold}')
        plt.title(f'Combined {metric_title}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_title)
        plt.legend()
        plt.tight_layout()

        pdf_path = os.path.join(RESULT_DIR, f"combined_{col}.pdf")
        svg_path = os.path.join(RESULT_DIR, f"combined_{col}.svg")
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        plt.close()


# =====================
# 工具：統計資料片段數
# =====================

def count_chunks_in_folder(folder_path, max_seq_len=MAX_SEQ_LEN):
    total_chunks = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            num_rows = len(df)
            chunks = num_rows // max_seq_len
            total_chunks += chunks
    return total_chunks


def best_result():
    results_csv_path = os.path.join(RESULT_DIR, "overall_experiment_log.csv")
    if not os.path.exists(results_csv_path):
        print("未發現網格搜尋結果 CSV，請先執行網格搜尋。")
        return None
    df = pd.read_csv(results_csv_path)
    best_row = df.loc[df['post_train_acc'].idxmax()]
    return best_row


# =====================
# 執行入口（網格實驗對齊原本架構）
# =====================
if __name__ == '__main__':
    if CLEAR_RESULT_DIR_BEFORE_RUN:
        print(f"[DEBUG] Clearing {RESULT_DIR} before run...")
        shutil.rmtree(RESULT_DIR, ignore_errors=True)
        os.makedirs(RESULT_DIR, exist_ok=True)

    # 與原檔案一致的網格
    batch_size_values = [16]
    learning_rate_values = [0.01]
    max_seq_len_values = [15]
    stride_values = [5]

    overall_results = []
    original_RESULT_DIR = RESULT_DIR

    for stride in stride_values:
        STRIDE = stride
        for bs in batch_size_values:
            for lr in learning_rate_values:
                for seq in max_seq_len_values:
                    # 覆寫全域
                    BATCH_SIZE = bs
                    LEARNING_RATE = lr
                    MAX_SEQ_LEN = seq

                    exp_id = f"bs_{bs}_lr_{lr}_seq_{seq}_stride_{stride}"
                    print(f"\n==== Running experiment: {exp_id} ====")
                    exp_result_dir = os.path.join(original_RESULT_DIR, exp_id)
                    os.makedirs(exp_result_dir, exist_ok=True)
                    RESULT_DIR = exp_result_dir

                    # 載入資料（會依目前 MAX_SEQ_LEN 切片）
                    all_sequences, all_labels = load_data()

                    # K-fold 訓練
                    final_metrics_df, all_folds_metrics = kfold_training(all_sequences, all_labels)

                    # 疊圖（選配，與原版一致）
                    plot_overlaid_metrics(all_folds_metrics)

                    # 儲存本次實驗的最終指標 log
                    log_csv_path = os.path.join(RESULT_DIR, "final_metrics.csv")
                    final_metrics_df.to_csv(log_csv_path, index=False)
                    print(f"Final metrics logged at: {log_csv_path}")

                    # 蒐集 overall 總結
                    overall_results.append({
                        'experiment_id': exp_id,
                        'batch_size': bs,
                        'learning_rate': lr,
                        'max_seq_len': seq,
                        'pre_train_loss': final_metrics_df['Pre-train Loss'].mean(),
                        'pre_train_acc': final_metrics_df['Pre-train Accuracy'].mean(),
                        'post_train_loss': final_metrics_df['Post-train Loss'].mean(),
                        'post_train_acc': final_metrics_df['Post-train Accuracy'].mean(),
                        'precision': final_metrics_df['Precision'].mean(),
                        'recall': final_metrics_df['Recall'].mean(),
                        'f1_score': final_metrics_df['F1-Score'].mean(),
                    })

                    # 還原 RESULT_DIR
                    RESULT_DIR = original_RESULT_DIR

    # 寫出 overall 總結
    overall_log_path = os.path.join(RESULT_DIR, "overall_experiment_log.csv")
    overall_df = pd.DataFrame(overall_results)
    overall_df.to_csv(overall_log_path, index=False)
    print(f"Overall experiment log saved at: {overall_log_path}")
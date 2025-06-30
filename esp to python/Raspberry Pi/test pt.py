import torch

# 載入 TorchScript
model = torch.jit.load("0626_fold_5_model_scripted.pt")
model.eval()

print(model)

# 測試推論
example_input = torch.rand(1, 10, 4)  # (batch_size, seq_len, input_dim)
with torch.no_grad():
    output = model(example_input)

print("✅ 推論成功，輸出：", output)
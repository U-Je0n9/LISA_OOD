import torch

local_model_weights = '/workspace/LISA/runs/lisa-7b/pytorch_model.bin'

# 로컬 가중치 로드
state_dict = torch.load(local_model_weights, map_location="cpu")

# 키 이름 변환: base_model.model 제거
new_state_dict = {}
prefix_to_remove = "base_model.model."

for key, value in state_dict.items():
    if key.startswith(prefix_to_remove):
        new_key = key[len(prefix_to_remove):]  # 접두사 제거
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value  # 접두사가 없는 경우 그대로 유지

print("save start")
new_weights_path = "/workspace/LISA/runs/lisa-7b/modified_pytorch_model.bin"
torch.save(new_state_dict, new_weights_path)

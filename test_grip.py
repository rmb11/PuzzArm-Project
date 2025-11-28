import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from arm import set_state, get_joints, apply_deltas

class GripNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(in_feats + 6, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, img, joints):
        feats = self.backbone(img)
        x = torch.cat([feats, joints], dim=1)
        return self.fc(x)

device = torch.device("cpu")
print("Using device:", device)

model = GripNet().to(device)
state_dict = torch.load("grip_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def run_attempt(attempt_num):
    print(f"\n===== ATTEMPT {attempt_num} =====")

    set_state("ready_to_grab")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error.")
        return
    print("Position puzzle under gripper, then press any key...")
    _, frame = cap.read()
    cv2.imshow(f"Attempt {attempt_num} - press any key", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_t = transform(img).unsqueeze(0).to(device)

    joints_list = get_joints(degrees=True)
    joints_t = torch.tensor(joints_list, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_deltas = model(img_t, joints_t).squeeze(0).cpu().numpy()

    print(f"Current joints:   {joints_list}")
    print(f"Predicted deltas: {pred_deltas}")

    apply_deltas(pred_deltas, scale=0.2, duration_ms=1500)

    print(f"✔ Attempt {attempt_num} complete")

def main():
    print("Starting grip test (3 attempts)")
    for i in range(1, 4):
        run_attempt(i)
        input("Reposition puzzle and press ENTER to continue...")
    print("\nFinished all attempts")

if __name__ == "__main__":
    main()
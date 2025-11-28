# PuzzArm Project – MVP Imitation Learning Demo

This project is a proof-of-concept to show how AI and robotics can work together using imitation learning. I used a webcam and an xArm1S robotic arm to collect my own dataset, trained a small model (GripNet) using ResNet18 in Google Colab and then tested the model using a Python script on my laptop.

The arm only moved slightly in the demo, but it repeated the behaviour both at home and in class which showed that the model did learn something from the joint examples I trained on. This meets my MVP goal.

---

## Included in this Repository

- `template.md` – Completed CRISP-DM document (main assessment)
- `grip_training.ipynb` – Training notebook (Google Colab)
- `test_grip.py` – Script used to run the live test on the robot
- `arm.py` – Basic arm control functions
- `camera.py` – Camera script
- `grip_model.pth` – Trained GripNet weights ([here](https://tafewa-my.sharepoint.com/:u:/r/personal/20123061_tafe_wa_edu_au/Documents/grip_model.zip?csf=1&web=1&e=L19JHE)
- `grip_dataset/` – Small sample of collected dataset images
- Samples of grip dataset
---

## Demo Video

Live robot test frrom home demonstrating model output:  
**[Watch Demo](https://www.youtube.com/watch?v=qkX_V8NIIWA)**

---

## Tools Used

- Python, PyTorch, OpenCV, Google Colab  
- CNN model (ResNet18 → GripNet regression)  
- Hiwonder xArm1S robotic arm + webcam

---

##  How I ran the test

```bash
python test_grip_model.py

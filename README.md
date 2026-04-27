# Medical Image Classification System
**JIIT Noida | AI & ML Lab | 3rd Semester**
Team: Ishita Arora | Priyani Rajvanshi | Hrishita Raj Singh | Samman Singh

---

## Project Structure
```
medical_classification/
│
├── main.py              ← Entry point (run this)
├── router.py            ← Routes image to correct model
├── train.py             ← Training script for all 3 models
├── requirements.txt
│
├── models/
│   ├── chest_model.py   ← Model 1: Pneumonia detection
│   ├── brain_model.py   ← Model 2: Brain tumor classification
|   |── eye_model.py     ← Model 3: Diabetic Retinopathy Detection
│   └── fracture_model.py← Model 4: Fracture detection
│
├── data/
│   ├── chest_xray/      ← Kaggle: paultimothymooney/chest-xray-pneumonia
│   ├── brain_mri/       ← Kaggle: masoudnickparvar/brain-tumor-mri-dataset
│   ├── retina_scan/     ← Kaggle: sovitrath/diabetic-retinopathy-2015-data-colored-resized
│   └── fracture/        ← Kaggle: pkdarabi/bone-fracture-detection-...
│
└── results/             ← Trained .h5 models saved here
```

---

## How to Run (Right Now - Dummy Mode)
```bash
pip install -r requirements.txt
python main.py
```

## How to Train (After downloading datasets)
```bash
python train.py --model chest       # Train chest model only
python train.py --model brain       # Train brain model only
python train.py --model fracture    # Train fracture model only
python train.py --model all         # Train all 3
```

---

## Datasets (Download from Kaggle)
| Model    | Dataset Link |
|----------|-------------|
| Chest    | kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Brain    | kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| Fracture | kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project |
| Eye      | https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized?select=colored_images |

Place downloaded data inside the `data/` folder matching the structure above.

---

## Status
- [x] Project structure
- [x] Router logic
- [x] Dummy predictions (working)
- [ ] Chest model training
- [ ] Brain model training  
- [ ] Fracture model training
- [ ] Evaluation & metrics

---

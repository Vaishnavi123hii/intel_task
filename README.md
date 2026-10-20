# Plz Readme!

## How I Approached the Task

After reading the problem statement I wasn't sure where to start. It took me about five hours to think of opening the starter kit, which had three files: a data downloader, a low-accuracy baseline model, and a verification script. I started by improving that baseline's accuracy.

That led me down a rabbit hole of 3Blue1Brown and Green Code lectures, trying to understand what I was actually building. I picked up scikit-learn, TensorFlow and PyTorch for the first time during this task. 

The code is AI-written, but I understand it and the architecture behind it.

Two days before the deadline I had a "finished" model, but the training data felt too small. Following PranavBhatP's bhaiya's suggestion, I looked into data augmentation, and this is the final model.

### References

- [Green Code](https://youtu.be/cAkMcPfY_Ns?si=XfCztcBISr1ApZhp)
- [3Blue1Brown](https://youtu.be/aircAruvnKk?si=_Ly_xSsnwDspkqNy)

## About

Lightweight CNN (depthwise-separable convolutions) that classifies waste images into 6 classes: cardboard, glass, metal, paper, plastic, trash. Exports to ONNX.

## Data

Folder layout (class subfolders, `ImageFolder` style):

```
submission/
├── download_data.py     
├── train.py              
├── requirements.txt     
├── model.onnx           
└── README.md
data/public/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

Train and val must contain the same class folders.

## Train

```bash
python train.py --epochs 60 --batch_size 32 --lr 0.003
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/public` | Dataset root |
| `--epochs` | `60` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.003` | Peak learning rate (OneCycle) |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--output` | `model.onnx` | ONNX output path |
| `--seed` | `42` | Random seed |

## Output

- `model.onnx`: the best checkpoint by validation accuracy (input `1x3x96x96`, dynamic batch)
- Constraints: under 350,000 parameters and under 1.5 MB (both printed during the run)

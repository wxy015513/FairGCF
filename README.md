# FairGCF
## requirements
torch==1.4.0
pandas==0.24.2
scipy==1.3.0
numpy==1.22.0
tensorboardX==1.8
scikit-learn==0.23.2
tqdm==4.48.2

## An example to run a 3-layer LightGCN

run FairGCF on AKindle dataset:
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="AKindle" --topks="[20]" --recdim=64

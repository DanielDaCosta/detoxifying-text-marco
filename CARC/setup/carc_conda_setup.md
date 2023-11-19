```
mkdir -p models/toxic
mkdir logs
```

```
module load conda
conda create --name marco python=3.10
eval "$(conda shell.bash hook)"

conda activate marco

pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
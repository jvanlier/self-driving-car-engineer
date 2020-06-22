#!/usr/bin/env /bin/bash
# Train many models in order to compare results later in MLflow
./model.py --epochs 200 --lr 0.01   --dropout 0.5
./model.py --epochs 200 --lr 0.001  --dropout 0.5
./model.py --epochs 200 --lr 0.0001 --dropout 0.5

./model.py --epochs 200 --lr 0.01   --dropout 0.3
./model.py --epochs 200 --lr 0.001  --dropout 0.3
./model.py --epochs 200 --lr 0.0001 --dropout 0.3

./model.py --epochs 200 --lr 0.01   --dropout 0.7
./model.py --epochs 200 --lr 0.001  --dropout 0.7
./model.py --epochs 200 --lr 0.0001 --dropout 0.7

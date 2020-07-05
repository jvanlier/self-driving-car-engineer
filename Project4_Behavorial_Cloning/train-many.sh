#!/usr/bin/env /bin/bash
# Train many models in order to compare results later in MLflow
EXPERIMENT="v6_mse_crop_moredata_flip"
set -exu

./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.01   --dropout 0.5
./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.001  --dropout 0.5
./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.0001 --dropout 0.5

./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.01   --dropout 0.3
./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.001  --dropout 0.3
./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.0001 --dropout 0.3

./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.01   --dropout 0.7
./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.001  --dropout 0.7
./model.py --exp ${EXPERIMENT} --epochs 200 --lr 0.0001 --dropout 0.7

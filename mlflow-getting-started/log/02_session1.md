python scripts/train.py --epochs 5 --learning_rate 0.01 --batch_size 64
Entrenando con 5 epochs, LR=0.01, Batch size=64
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.91M/9.91M [03:32<00:00, 46.6kB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28.9k/28.9k [00:04<00:00, 5.96kB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.65M/1.65M [00:29<00:00, 55.0kB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 2.26MB/s]
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64

SimpleNN(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_stack): Sequential(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=10, bias=True)
  )
)

inputs:
  [Tensor('float32', (-1, 28, 28))]
outputs:
  [Tensor('float32', (-1, 10))]
params:
  None

Signature: inputs:
  [Tensor('float32', (-1, 28, 28))]
outputs:
  [Tensor('float32', (-1, 10))]
params:
  None

D:\Anaconda_Envs\nlp\Lib\site-packages\mlflow\tracking\_tracking_service\utils.py:184: FutureWarning: The filesystem tracking backend (e.g., './mlruns') is deprecated as of February 2026. Consider transitioning to a database backend (e.g., 'sqlite:///mlflow.db') to take advantage of the latest MLflow features. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance.
  return FileStore(store_uri, store_uri)
Epoch 1
-------------------------------
loss: 2.303609  [    0/60000]
loss: 2.259071  [ 6400/60000]
loss: 2.234816  [12800/60000]
loss: 2.055236  [19200/60000]
loss: 1.985016  [25600/60000]
loss: 1.761873  [32000/60000]
loss: 1.360176  [38400/60000]
loss: 1.312932  [44800/60000]
loss: 0.985676  [51200/60000]
loss: 0.783107  [57600/60000]
Test Error:
 Accuracy: 82.4%, Avg loss: 0.736797

Epoch 2
-------------------------------
loss: 0.812831  [    0/60000]
loss: 0.604257  [ 6400/60000]
loss: 0.643249  [12800/60000]
loss: 0.536703  [19200/60000]
loss: 0.487606  [25600/60000]
loss: 0.491869  [32000/60000]
loss: 0.354943  [38400/60000]
loss: 0.581218  [44800/60000]
loss: 0.477559  [51200/60000]
loss: 0.475699  [57600/60000]
Test Error:
 Accuracy: 88.5%, Avg loss: 0.408056

Epoch 3
-------------------------------
loss: 0.446269  [    0/60000]
loss: 0.337377  [ 6400/60000]
loss: 0.373561  [12800/60000]
loss: 0.405710  [19200/60000]
loss: 0.343081  [25600/60000]
loss: 0.391784  [32000/60000]
loss: 0.244637  [38400/60000]
loss: 0.473858  [44800/60000]
loss: 0.394076  [51200/60000]
loss: 0.434502  [57600/60000]
Test Error:
 Accuracy: 90.0%, Avg loss: 0.343558

Epoch 4
-------------------------------
loss: 0.335350  [    0/60000]
loss: 0.284300  [ 6400/60000]
loss: 0.284011  [12800/60000]
loss: 0.371730  [19200/60000]
loss: 0.294200  [25600/60000]
loss: 0.351568  [32000/60000]
loss: 0.207612  [38400/60000]
loss: 0.427970  [44800/60000]
loss: 0.348912  [51200/60000]
loss: 0.411088  [57600/60000]
Test Error:
 Accuracy: 90.8%, Avg loss: 0.312603

Epoch 5
-------------------------------
loss: 0.276137  [    0/60000]
loss: 0.264876  [ 6400/60000]
loss: 0.237224  [12800/60000]
loss: 0.354266  [19200/60000]
loss: 0.262333  [25600/60000]
loss: 0.324267  [32000/60000]
loss: 0.187417  [38400/60000]
loss: 0.398939  [44800/60000]
loss: 0.314776  [51200/60000]
loss: 0.387513  [57600/60000]
Test Error:
 Accuracy: 91.6%, Avg loss: 0.290802

Modelo guardado en model.pth
Done!
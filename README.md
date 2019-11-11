# Pytorch MNIST Example custumed with SofNDLTeches

This is cloned official Pytorch MNIST example code and it's modified with the techniques from the following website.  
[How we beat the FastAI leaderboard score by +19.77%…a synergy of new deep learning techniques for your consideration.](https://medium.com/@lessw/how-we-beat-the-fastai-leaderboard-score-by-19-77-a-cbb2338fab5c)  

## Additional Features
- [x] Global Average Pooling.
- [x] Mish activate function.
- [x] Self Attention module.
- [x] Flat plus Cosine Annealing LR Scheduler.
- [x] Ranger optimizer.
- [x] Label Smoothing Loss
- [x] TensorBoard

## Usage
```bash
pip install -r requirements.txt
python main_custom.py
```  

To check training, after run.  
```
tensorboard --logdir runs
```

## Options

```
 ❯ python main_custom.py -h
usage: main_custom.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                      [--lr LR] [--momentum M] [--no-cuda] [--seed S]
                      [--log-interval N] [--save-model]
                      [--optimizer {raner,ranerqh,sgd}] [--sa] [--mish]
                      [--smooth SMOOTH] [--gp] [--fpa]

PyTorch MNIST Example

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.01)
  --momentum M          SGD momentum (default: 0.5)
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training
                        status
  --save-model          For Saving the current Model
  --optimizer {raner,ranerqh,sgd}
                        choose optimizer from choices
  --sa                  use self attention module
  --mish                use Mish activate function
  --smooth SMOOTH       put float value to label smooth, default:sce
  --gp                  use global pooling
  --fpa                 use Flat plus annealing scheduler
```

## Reference
- [Ranger-Mish-ImageWoof-5](https://github.com/lessw2020/Ranger-Mish-ImageWoof-5)
- [res2net-plus](https://github.com/lessw2020/res2net-plus) 
- [Ranger-Deep-Learning-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)

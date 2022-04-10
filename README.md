# Post-processing for Individual Fairness 

![](http://petersen.ai/images/if-post-processing-logo-low.png)

Official implementation for our NeurIPS 2021 Paper "Post-processing for Individual Fairness".

Paper @ [ArXiv](https://arxiv.org/pdf/2110.13796.pdf),
Video @ [Youtube](https://www.youtube.com/watch?v=9PyKODDewPA).


## 👩‍💻 Setup

We recommend creating a virtual environment under Python 3.6 as follows:

```shell script
virtualenv -p python3.6 .env1
. .env1/bin/activate
pip install -r requirements.txt
pip install cvxpy
```

The code runs on Linux / macOS on CPU and does not require a GPU. 
For installing `cvxpy`, CMake 3.2 or higher is required. 
On some machines `CFLAGS=-std=c99 pip install cvxpy` fixes a compiler error.

## 📚 Data Sets

The sentiment data set is included in the repository.
For the Bios and Toxicity data sets, downloading the respective BERT embeddings is necessary.
Each of them has a size of around **40GB** and can be downloaded and extracted with the following commands:

```shell
# Warning: each of the large data sets has a size of around 40GB:
wget https://publicdata1.nyc3.digitaloceanspaces.com/IF_Bios_BERT.tar.gz
tar -xvzf IF_Bios_BERT.tar.gz && rm IF_Bios_BERT.tar.gz

wget https://publicdata1.nyc3.digitaloceanspaces.com/IF_Toxicity_BERT.tar.gz
tar -xvzf IF_Toxicity_BERT.tar.gz && rm IF_Toxicity_BERT.tar.gz
```

## 🧫 Running the Experiments

To reproduce the sentiment experiments with the closed form GLIF(-NRW), run `run_sentiment.py`:

```shell
python -u run_sentiment.py -ni 2_000 --nloglr 3 --seed 0 --tau 30 --no_cvx
```
Removing `--no_cvx` will include the (slow) CVXPY IF-constraints method.
Removing `--tau 30` will run it for a range of possible taus, which also does not take significantly longer.

### 🧪 Coordinate Descent Experiments

`run_coordinate_descent.py` can run the coordinate descent method on all 3 data sets.
Note that, as it handles all data sets and has an overhead for handling such large data, this code is more complex than 
the simple sentiment experiment with the closed form (`run_sentiment.py`). We recommend checking out `run_sentiment.py`
first.

To run `run_coordinate_descent.py`, you can use the following commands:

```shell
# sentiment
python -u run_coordinate_descent.py -ni 2_000 --nloglr 3 --seed 0 --dataset sentiment --lambda_GLIF .1 --lambda_GLIF_NRW .1 --tau 30

# bios
python -u run_coordinate_descent.py -ni 10_000 --nloglr 5 --seed 0 --dataset bios --lambda_GLIF 10 --lambda_GLIF_NRW .1 --tau 16 --test_fraction 0.1

# toxicity
python -u run_coordinate_descent.py -ni 10_000 --nloglr 3 --seed 0 --dataset toxicity --lambda_GLIF 30 --lambda_GLIF_NRW .1 --tau .4 --test_fraction 0.05
```

Note that running on bios and toxicity can take a long time, therefore reducing `--test_fraction` to 0.01 or 0.001 
significantly speeds up the code.
Again, omitting `--tau` will run a range of adequate taus for each data set.



## 📖 Citing

```bibtex
@article{petersen2021post,
  title={Post-processing for Individual Fairness},
  author={Petersen, Felix and Mukherjee, Debarghya and Sun, Yuekai and Yurochkin, Mikhail},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## ⚖ License

The code is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.




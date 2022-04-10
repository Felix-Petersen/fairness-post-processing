# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import argparse
import random
from tqdm import tqdm, trange
import time
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pprint import pprint
from utils_ import load_data, load_test_names, load_nyc_names


########################################################################################################################
"""

python -u run_sentiment.py -ni 2000 --nloglr 3 --seed 0 
python -u run_sentiment.py -ni 2000 --nloglr 3 --seed 0 --tau 30

python -u run_sentiment.py -ni 2000 --nloglr 3 --seed 0 --no_cvx
python -u run_sentiment.py -ni 2000 --nloglr 3 --seed 0 --no_cvx --tau 30

"""
########################################################################################################################


parser = argparse.ArgumentParser(description='')

parser.add_argument('-ni', '--num_iterations', type=int, default=4_000)
parser.add_argument('--batch_size', type=int, default=1_000)
parser.add_argument('--nloglr', type=float, default=3.)

parser.add_argument('--dataset', default='sentiment', type=str, choices=['sentiment'])

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--lambda_GLIF', type=float, default=0.1)
parser.add_argument('--lambda_GLIF_NRW', type=float, default=0.1)
parser.add_argument('--theta', type=float, default=1e-4)
parser.add_argument('--taus', nargs='+', type=float, default=list(np.logspace(np.log10(10), np.log10(100), 51)))
parser.add_argument('--Ls', nargs='+', type=float, default=[1.5, 1.7, 1.8, 1.9, 2.0, 2.25, 2.5, 2.75, 3., 3.5, 4.])

parser.add_argument('--no_cvx', action='store_true')

args = parser.parse_args()

print(vars(args))

torch.set_num_threads(min(8, torch.get_num_threads()))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cpu')


########################################################################################################################
# Training #############################################################################################################
########################################################################################################################

def sample_batch_idx(y, n_per_class):
    batch_idx = []
    for i in range(y.shape[1]):
        batch_idx += np.random.choice(np.where(y[:, i] == 1)[0], size=n_per_class, replace=False).tolist()

    np.random.shuffle(batch_idx)
    return batch_idx


def get_model():
    input_dim = {
        'sentiment': 300,
        'bios': 768,
        'toxicity': 768,
    }[args.dataset]
    hidden_dim = 1_000 if args.dataset == 'sentiment' else 2_000
    output_dim = {
        'sentiment': 2,
        'bios': 28,
        'toxicity': 2,
    }[args.dataset]
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    ).to(device)


########################################################################################################################
#  Basic Evaluation  ###################################################################################################
########################################################################################################################


def eval_model(main_pred=None, pred_for_fairness=None, y_test_local_sentiment=None, name='unknown'):
    if main_pred is None:
        main_pred = model(torch.tensor(X_test).float().to(device))

    if pred_for_fairness is None:
        pred_for_fairness = model(torch.tensor(test_names_embed).float().to(device))
        pred_for_fairness = pred_for_fairness[:, 1] - pred_for_fairness[:, 0]

    # --------

    if args.dataset == 'sentiment':
        assert len(pred_for_fairness.shape) == 1, pred_for_fairness.shape

        acc = (main_pred.argmax(1) == torch.tensor(
            y_test if y_test_local_sentiment is None else y_test_local_sentiment
        ).float().to(device).argmax(1)).float().mean()

        test_df['naive_logits'] = pred_for_fairness.cpu().detach().numpy()
        race_gap = (
                test_df[test_df['race'] == 'White']['naive_logits'].mean()
                - test_df[test_df['race'] == 'Black']['naive_logits'].mean()
        )
        gender_gap = (
                test_df[test_df['gender'] == 'Female']['naive_logits'].mean()
                - test_df[test_df['gender'] == 'Male']['naive_logits'].mean()
        )

        pprint({
            'test_acc_{}'.format(name): acc.item(),
            'race_gap_{}'.format(name): race_gap.item(),
            'gender_gap_{}'.format(name): gender_gap.item(),
            'logit_std_{}'.format(name): pred_for_fairness.std().item(),
        })


########################################################################################################################
#  Main  ###############################################################################################################
########################################################################################################################


if __name__ == '__main__':

    ####################################################################################################################
    #  Sentiment w/ Names  #############################################################################################
    ####################################################################################################################

    if args.dataset == 'sentiment':
        # The files in this repo are the respective relevant files (or relevant parts of the embedding) from:
        # http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
        # http://nlp.stanford.edu/data/glove.42B.300d.zip

        data_path = './project_data/'
        embeddings_path = './project_data/sentiment_glove.42B.300d.txt'
        nyc_names_path = './project_data/Popular_Baby_Names.csv'

        # Load data and embeddings
        # Loading GloVe might take couple of minutes
        embeddings, X_train, X_test, y_train, y_test, train_vocab, test_vocab = load_data(data_path, embeddings_path,
                                                                                          state=args.seed)

        # Load test names and their embeddings
        test_df, test_names_embed = load_test_names(embeddings)

        # Load Popular NYC Baby Names and their embeddings
        nyc_names_embed = load_nyc_names(nyc_names_path, embeddings)
        prohibited_subspace = nyc_names_embed

    else:
        raise NotImplementedError(args.dataset)

    num_test_samples = X_test.shape[0]
    print('num_train_word', X_train.shape[0])
    print('num_test_samples', num_test_samples)
    print('test_set_size', X_test.shape[0], test_names_embed.shape[0], X_test.shape[0] + test_names_embed.shape[0])

    ####################################################################################################################
    #  Prohibited Subspace  ############################################################################################
    ####################################################################################################################

    # Learning sensitive direction from Popular Baby Names
    tSVD = TruncatedSVD(n_components={
        'sentiment': 50,
        'bios': 25,
        'toxicity': 25,
    }[args.dataset])
    tSVD.fit(prohibited_subspace)
    svd_sens_directions = tSVD.components_
    svd_sens_directions = torch.tensor(svd_sens_directions).float().to(device)
    print(svd_sens_directions.shape)

    ####################################################################################################################
    #  Baseline Model  #################################################################################################
    ####################################################################################################################

    print('')
    print('Baseline Model')
    print('')

    model = get_model()
    optim = torch.optim.Adam(model.parameters(), lr=10**(-args.nloglr))

    for iter_idx in trange(args.num_iterations):
        batch_idx = sample_batch_idx(y_train, args.batch_size // (y_train.shape[1]))

        batch_x = torch.tensor(X_train[batch_idx]).float().to(device)
        batch_y = torch.tensor(y_train[batch_idx]).float().to(device)

        y_pred = model(batch_x)
        loss = torch.nn.CrossEntropyLoss()(y_pred, batch_y.argmax(1))

        acc = (y_pred.argmax(1) == batch_y.argmax(1)).float().mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    print('train loss / acc', loss.item(), acc.item() * 100., '%')
    eval_model(name='baseline')

    ####################################################################################################################
    #  Convex Solver Evaluation  #######################################################################################
    ####################################################################################################################

    if not args.no_cvx:

        import cvxpy as cp

        print('')
        print('Convex Solver Evaluation')
        print('')

        with torch.no_grad():

            X_test_local = torch.tensor(X_test).float().to(device)[::]
            # y_test_local = (torch.tensor(y_test[:, 1]) == 1).long()[::3]
            y_test_local = y_test[::]
            data = torch.cat([
                X_test_local,
                torch.tensor(test_names_embed).float().to(device),
            ])
            num_test_samples_local = X_test_local.shape[0]

            outputs = model(data)

            basis = svd_sens_directions.cpu().numpy().T
            proj = np.linalg.inv(np.matmul(basis.T, basis))
            proj = np.matmul(basis, proj)
            proj = np.matmul(proj, basis.T)
            proj_compl = np.eye(proj.shape[0]) - proj
            proj_compl = torch.tensor(proj_compl).float().to(device)

            fair_space_data = data @ proj_compl
            fair_space_data = fair_space_data.cpu().numpy()


            def e_dist(A, B, cosine=False, eps=1e-10):
                A_n = (A ** 2).sum(axis=1).reshape(-1, 1)
                B_n = (B ** 2).sum(axis=1).reshape(1, -1)
                inner = np.matmul(A, B.T)
                if cosine:
                    return 1 - inner / (np.sqrt(A_n * B_n) + eps)
                else:
                    return A_n - 2 * inner + B_n


            fair_space_data_squared_distances = e_dist(fair_space_data, fair_space_data)
            fair_space_data_squared_distances = torch.relu(
                torch.tensor(fair_space_data_squared_distances).to(device))
            fair_space_data_distances = fair_space_data_squared_distances.sqrt().cpu().detach().numpy()

            for L in args.Ls:

                print('L', L)

                logits = outputs[:, 1] - outputs[:, 0]
                logits = logits.unsqueeze(-1).cpu().detach().numpy()

                t_s = time.time()

                y_hat = cp.Variable((logits.shape[0], 1))
                ones = np.ones((1, logits.shape[0]))

                constraints = [
                    y_hat @ ones - (y_hat @ ones).T - np.eye(logits.shape[0]) <= L * fair_space_data_distances]
                objective = cp.Minimize(cp.sum((y_hat - logits) ** 2))

                prob = cp.Problem(objective, constraints)
                result = prob.solve(solver='SCS')

                y_updated = y_hat.value.reshape(-1)

                t_e = time.time()
                print('Time [s]: {:.3f}'.format(t_e - t_s))

                y_pred = torch.stack([
                    torch.zeros_like(torch.tensor(y_updated[:num_test_samples_local]).float().to(device)),
                    torch.tensor(y_updated[:num_test_samples_local]).float().to(device)
                ], dim=1)
                eval_model(
                    y_pred, torch.tensor(y_updated[num_test_samples_local:]),
                    y_test_local_sentiment=y_test_local, name='cvx_L_{}'.format(L)
                )

    ####################################################################################################################
    #  GLIF: Laplacian Evaluation (closed form)  #######################################################################
    ####################################################################################################################

    print('')
    print('GLIF: Laplacian Evaluation (closed form)')
    print('')

    with torch.no_grad():

        data = torch.cat([
            torch.tensor(X_test).float().to(device),
            torch.tensor(test_names_embed).float().to(device),
        ])

        outputs = model(data)

        basis = svd_sens_directions.cpu().numpy().T
        proj = np.linalg.inv(np.matmul(basis.T, basis))
        proj = np.matmul(basis, proj)
        proj = np.matmul(proj, basis.T)
        proj_compl = np.eye(proj.shape[0]) - proj
        proj_compl = torch.tensor(proj_compl).float().to(device)

        fair_space_data = data @ proj_compl
        fair_space_data = fair_space_data.cpu().numpy()

        def e_dist(A, B, cosine=False, eps=1e-10):
            A_n = (A ** 2).sum(axis=1).reshape(-1, 1)
            B_n = (B ** 2).sum(axis=1).reshape(1, -1)
            inner = np.matmul(A, B.T)
            if cosine:
                return 1 - inner / (np.sqrt(A_n * B_n) + eps)
            else:
                return A_n - 2 * inner + B_n

        fair_space_data_squared_distances = e_dist(fair_space_data, fair_space_data)
        fair_space_data_squared_distances = torch.relu(torch.tensor(fair_space_data_squared_distances).to(device))

        for tau in args.taus:

            print('tau', tau)

            t_s = time.time()

            fair_similarity_W = torch.exp(-fair_space_data_squared_distances * args.theta) * \
                                (fair_space_data_squared_distances <= tau).float()
            D_ii = torch.diag_embed(fair_similarity_W.sum(1))
            L = D_ii - fair_similarity_W

            y_updated = torch.inverse(args.lambda_GLIF * L + torch.eye(L.shape[0]).to(device)) @ outputs

            t_e = time.time()
            print('Time [s]: {:.3f}'.format(t_e - t_s))

            names_logits = y_updated[num_test_samples:]
            names_logits = names_logits[:, 1] - names_logits[:, 0]

            eval_model(y_updated[:num_test_samples], names_logits, name='laplacian_tau_{}'.format(tau))

    ####################################################################################################################
    #  GLIF-NRW: Normalized Random Walk Laplacian Evaluation (closed form)  ############################################
    ####################################################################################################################

    print('')
    print('GLIF-NRW: Normalized Random Walk Laplacian Evaluation (closed form)')
    print('')

    with torch.no_grad():

        data = torch.cat([
            torch.tensor(X_test).float().to(device),
            torch.tensor(test_names_embed).float().to(device),
        ])

        outputs = model(data)

        basis = svd_sens_directions.cpu().numpy().T
        proj = np.linalg.inv(np.matmul(basis.T, basis))
        proj = np.matmul(basis, proj)
        proj = np.matmul(proj, basis.T)
        proj_compl = np.eye(proj.shape[0]) - proj
        proj_compl = torch.tensor(proj_compl).float().to(device)

        fair_space_data = data @ proj_compl
        fair_space_data = fair_space_data.cpu().numpy()

        def e_dist(A, B, cosine=False, eps=1e-10):
            A_n = (A ** 2).sum(axis=1).reshape(-1, 1)
            B_n = (B ** 2).sum(axis=1).reshape(1, -1)
            inner = np.matmul(A, B.T)
            if cosine:
                return 1 - inner / (np.sqrt(A_n * B_n) + eps)
            else:
                return A_n - 2 * inner + B_n

        fair_space_data_squared_distances = e_dist(fair_space_data, fair_space_data)
        fair_space_data_squared_distances = torch.relu(torch.tensor(fair_space_data_squared_distances).to(device))

        for tau in args.taus:

            print('tau', tau)

            t_s = time.time()

            fair_similarity_W = torch.exp(-fair_space_data_squared_distances * args.theta) * \
                                (fair_space_data_squared_distances <= tau).float()
            D_ii = torch.diag_embed(fair_similarity_W.sum(1))
            D_ii_to_minus_half = torch.diag_embed(fair_similarity_W.sum(1).pow(-.5))
            W_tilde = D_ii_to_minus_half @ fair_similarity_W @ D_ii_to_minus_half
            D_tilde_to_minus_one = torch.diag_embed(W_tilde.sum(1).pow(-1))
            W = D_tilde_to_minus_one @ W_tilde
            L = torch.eye(D_ii.shape[0]).to(device) - W
            L = (L.T + L) / 2

            avg_degree = fair_similarity_W.sum(1).mean()
            print('avg_degree', avg_degree.item())

            y_updated = torch.inverse(args.lambda_GLIF_NRW*avg_degree * L + torch.eye(L.shape[0]).to(device)) @ outputs

            t_e = time.time()
            print('Time [s]: {:.3f}'.format(t_e - t_s))

            names_logits = y_updated[num_test_samples:]
            names_logits = names_logits[:, 1] - names_logits[:, 0]

            eval_model(y_updated[:num_test_samples], names_logits, name='laplacian_random_walk_tau_{}'.format(tau))

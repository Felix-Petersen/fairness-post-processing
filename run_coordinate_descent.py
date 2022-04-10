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
from sklearn.model_selection import train_test_split
from pprint import pprint
from utils_ import load_data, load_test_names, load_nyc_names


########################################################################################################################
"""

wget https://publicdata1.nyc3.digitaloceanspaces.com/IF_Bios_BERT.tar.gz
tar -xvzf IF_Bios_BERT.tar.gz && rm IF_Bios_BERT.tar.gz

wget https://publicdata1.nyc3.digitaloceanspaces.com/IF_Toxicity_BERT.tar.gz
tar -xvzf IF_Toxicity_BERT.tar.gz && rm IF_Toxicity_BERT.tar.gz

python -u run_coordinate_descent.py -ni 2000 --nloglr 3 --seed 0 --dataset sentiment --lambda_GLIF .1 --lambda_GLIF_NRW .1 --tau 30

python -u run_coordinate_descent.py -ni 10_000 --nloglr 5 --seed 0 --dataset bios --lambda_GLIF 10 --lambda_GLIF_NRW .1 --tau 16 --test_fraction 0.1

python -u run_coordinate_descent.py -ni 10_000 --nloglr 3 --seed 0 --dataset toxicity --lambda_GLIF 30 --lambda_GLIF_NRW .1 --tau .4 --test_fraction 0.05

"""
########################################################################################################################


parser = argparse.ArgumentParser(description='')

parser.add_argument('-ni', '--num_iterations', type=int, default=2_000)
parser.add_argument('--batch_size', type=int, default=1_000)
parser.add_argument('--nloglr', type=float, default=3.)

parser.add_argument('--dataset', default='sentiment', type=str, choices=['sentiment', 'bios', 'toxicity'])
parser.add_argument('--test_fraction', default=None, type=float)

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--lambda_GLIF', type=float, required=True)
parser.add_argument('--lambda_GLIF_NRW', type=float, required=True)
parser.add_argument('--taus', nargs='+', type=float, default=[])

parser.add_argument('--coo_epochs', type=int, default=10)

args = parser.parse_args()

print(vars(args))

torch.set_num_threads(min(8, torch.get_num_threads()))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cpu')

if args.taus:
    taus = args.taus
else:
    taus = {
        'sentiment': list(np.logspace(np.log10(10), np.log10(100), 51)),
        'toxicity': list(np.logspace(np.log10(.1), np.log10(1), 51)),
        'bios': list(np.logspace(np.log10(1), np.log10(100), 51)),
    }[args.dataset]


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
        if args.dataset == 'sentiment':
            pred_for_fairness = model(torch.tensor(test_names_embed).float().to(device))
            pred_for_fairness = pred_for_fairness[:, 1] - pred_for_fairness[:, 0]
        elif args.dataset == 'bios':
            pred_for_fairness = model(torch.tensor(X_counter_test).float().to(device))
        elif args.dataset == 'toxicity':
            assert X_test_counter.shape[0] == 51, X_test_counter.shape
            assert X_test_counter.shape[2] == 768, X_test_counter.shape
            x = torch.tensor(X_test_counter).reshape(-1, 768).float().to(device)
            pred_for_fairness = model(x)
            pred_for_fairness = pred_for_fairness.reshape(51, X_test_counter.shape[1], 2)

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

    # --------

    elif args.dataset == 'bios':
        accs = (main_pred.argmax(1) == torch.tensor(y_test).float().to(device).argmax(1)).float()
        accs = [accs[y_test.argmax(1) == i].mean() for i in range(28)]
        acc = torch.mean(torch.stack(accs))

        consistency = (main_pred.argmax(1) == pred_for_fairness.argmax(1)).float().mean()

        pprint({
            'test_acc_{}'.format(name): acc.item(),
            'test_consistency_{}'.format(name): consistency.item(),
        })

    # --------

    elif args.dataset == 'toxicity':
        accs = (main_pred.argmax(1) == torch.tensor(y_test).float().to(device).argmax(1)).float()
        accs = [accs[y_test.argmax(1) == i].mean() for i in range(2)]
        acc = torch.mean(torch.stack(accs))

        fairness = (pred_for_fairness.argmax(2).float().mean(0) == 0.).float().mean() \
                   + (pred_for_fairness.argmax(2).float().mean(0) == 1.).float().mean()

        pprint({
            'test_acc_{}'.format(name): acc.item(),
            'test_consistency_{}'.format(name): fairness.item(),
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

        assert args.test_fraction is None, args.test_fraction

    ####################################################################################################################
    #  Bios  ###########################################################################################################
    ####################################################################################################################

    elif args.dataset == 'bios':
        # wget https://publicdata1.nyc3.digitaloceanspaces.com/IF_Bios_BERT.tar.gz
        # tar -xvzf IF_Bios_BERT.tar.gz && rm IF_Bios_BERT.tar.gz

        bios_train_size = 0.98 if args.test_fraction is None else 1 - args.test_fraction

        bios_datafolder = './IF_Bios_BERT/'

        y_title = list(map(str, np.load(bios_datafolder + 'bios_titles.npy')))
        y_gender = list(map(str, np.load(bios_datafolder + 'bios_gedner.npy')))

        categories = ['dietitian', 'physician', 'photographer', 'dentist', 'surgeon', 'journalist',
                      'pastor', 'yoga_teacher', 'professor', 'accountant', 'architect', 'interior_designer',
                      'personal_trainer', 'chiropractor', 'poet', 'comedian', 'rapper', 'filmmaker',
                      'nurse', 'dj', 'painter', 'attorney', 'model', 'software_engineer', 'teacher',
                      'paralegal', 'composer', 'psychologist']

        from sklearn.preprocessing import OneHotEncoder

        one_hot_title = OneHotEncoder(sparse=False, categories=[categories])
        y_title = one_hot_title.fit_transform([[y] for y in y_title])
        one_hot_gender = OneHotEncoder(sparse=False, categories='auto')
        y_gender = one_hot_gender.fit_transform(np.array(y_gender).reshape(-1, 1))

        ## Partition data
        np.random.seed(args.seed)
        N = y_title.shape[0]
        bios_real = np.load(bios_datafolder + 'X_bert_real_seed_%d.npy' % (args.seed%10))
        bios_counter = np.load(bios_datafolder + 'X_bert_counter_seed_%d.npy' % (args.seed%10))
        idx_train = np.random.choice(N, int(N * bios_train_size), replace=False)
        idx_test = np.setdiff1d(range(N), idx_train)
        X_real_train, X_real_test = bios_real[idx_train], bios_real[idx_test]
        X_counter_train, X_counter_test = bios_counter[idx_train], bios_counter[idx_test]
        y_train, y_test, gender_train, gender_test = y_title[idx_train], y_title[idx_test], y_gender[idx_train], y_gender[
            idx_test]

        X_train, X_test = X_real_train, X_real_test

        prohibited_subspace = X_real_train - X_counter_train

    ####################################################################################################################
    #  Toxicity  #######################################################################################################
    ####################################################################################################################

    elif args.dataset == 'toxicity':
        # wget https://publicdata1.nyc3.digitaloceanspaces.com/IF_Toxicity_BERT.tar.gz
        # tar -xvzf IF_Toxicity_BERT.tar.gz && rm IF_Toxicity_BERT.tar.gz

        toxicity_test_size = 0.003 if args.test_fraction is None else args.test_fraction

        data_folder = './IF_Toxicity_BERT/'

        with open(data_folder + 'adjectives_people.txt', 'r') as f:
            IDENTITY_TERMS = np.array([w.strip() for w in f.readlines()])

        X_nid = np.load(data_folder + 'X_bert_nid_kaggle.npy')
        group_nid = np.load(data_folder + 'subgroups_nid.npy')
        target_nid = np.load(data_folder + 'target_nid.npy')
        target_nid = np.column_stack((1 * (target_nid == 0), 1 * (target_nid == 1)))

        X_id = np.load(data_folder + 'original_X_bert_id_kaggle.npy')
        group_id = np.load(data_folder + 'subgroups_id.npy')
        target_id = np.load(data_folder + 'target_id.npy')
        target_id = np.column_stack((1 * (target_id == 0), 1 * (target_id == 1)))

        id_mask = np.load(data_folder + 'id_mask.npy')
        n_nid = X_nid.shape[0]

        X_terms = []
        for term in IDENTITY_TERMS:
            X_term = np.load(data_folder + 'X_bert_kaggle_' + term + '.npy')
            X_terms.append(X_term)

        np.random.seed(args.seed)
        terms_train_idx = np.random.choice(len(IDENTITY_TERMS), size=25, replace=False)
        idx_nid_train, idx_nid_test = train_test_split(np.arange(X_nid.shape[0]), test_size=toxicity_test_size)
        idx_id_train, idx_id_test = train_test_split(np.arange(X_id.shape[0]), test_size=toxicity_test_size)

        id_train_mask = id_mask[idx_id_train][:, terms_train_idx]
        idx_train_INactive_id = idx_id_train[id_train_mask.sum(axis=1) == 0]
        idx_id_train = idx_id_train[id_train_mask.sum(axis=1) > 0]
        X_nid_seed = np.vstack((X_nid, X_id[idx_train_INactive_id]))
        target_nid_seed = np.vstack((target_nid, target_id[idx_train_INactive_id]))
        group_nid_seed = np.vstack((group_nid, group_id[idx_train_INactive_id]))
        idx_nid_train = np.concatenate((idx_nid_train, n_nid + np.arange(len(idx_train_INactive_id))))

        X_train = np.vstack((X_nid_seed[idx_nid_train], X_id[idx_id_train]))
        X_test = np.vstack((X_nid[idx_nid_test], X_id[idx_id_test]))
        y_train = np.vstack((target_nid_seed[idx_nid_train], target_id[idx_id_train]))
        y_test = np.vstack((target_nid[idx_nid_test], target_id[idx_id_test]))
        groups_train = np.vstack((group_nid_seed[idx_nid_train], group_id[idx_id_train]))
        groups_test = np.vstack((group_nid[idx_nid_test], group_id[idx_id_test]))
        X_test_counter = [X_id[idx_id_test]] + [X_terms_id[idx_id_test] for X_terms_id in X_terms]
        X_test_counter = np.array(X_test_counter)

        X_test_all = np.vstack((X_nid[idx_nid_test], X_test_counter.reshape(-1, 768)))
        y_test_all = np.vstack((
            target_nid[idx_nid_test],
            np.repeat(np.expand_dims(target_id[idx_id_test], 0), 51, axis=0).reshape(-1, 2)
        ))

        metric_X = [X_terms[i][idx_id_train] for i in terms_train_idx]
        terms_mean = np.mean(metric_X, axis=0)
        prohibited_subspace = np.vstack([X_t - terms_mean for X_t in metric_X])

    else:
        raise NotImplementedError(args.dataset)

    if args.dataset == 'sentiment':
        num_test_samples = X_test.shape[0]
    elif args.dataset == 'bios':
        num_test_samples = X_test.shape[0]
        print('test_set_size', X_real_test.shape[0] + X_counter_test.shape[0])
    elif args.dataset == 'toxicity':
        num_test_samples1 = len(idx_nid_test)
        num_test_samples2 = len(idx_id_test)
        num_test_samples = X_test.shape[0]
        assert num_test_samples1 + num_test_samples2 == num_test_samples, (num_test_samples1, num_test_samples2, num_test_samples)
        print('num_test_samples1', num_test_samples1)
        print('num_test_samples2', num_test_samples2)
        print('test_set_size', X_test_all.shape[0])

    print('num_test_samples', num_test_samples)
    print('num_train_samples', X_train.shape[0])

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
    #  GLIF: Laplacian Evaluation (coordinate descent)  ################################################################
    ####################################################################################################################

    print('')
    print('GLIF: Laplacian Evaluation (coordinate descent)')
    print('')

    with torch.no_grad():

        for tau in taus:

            print('tau', tau)

            if args.dataset == 'sentiment':
                data = torch.cat([
                    torch.tensor(X_test).float().to(device),
                    torch.tensor(test_names_embed).float().to(device),
                ])
            elif args.dataset == 'bios':
                data = torch.cat([
                    torch.tensor(X_test).float().to(device),
                    torch.tensor(X_counter_test).float().to(device),
                ])
            elif args.dataset == 'toxicity':
                data = torch.cat([
                    torch.tensor(X_test_all).float().to(device),
                ])
            else:
                raise NotImplementedError(args.dataset)

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

            def e_dist_torch(A, B):
                A_n = (A ** 2).sum(dim=1).reshape(-1, 1)
                B_n = (B ** 2).sum(dim=1).reshape(1, -1)
                inner = A @ B.T
                return A_n - 2 * inner + B_n

            fair_similarity_Ws = []
            fair_similarity_Ds = []

            t_s = time.time()

            for fair_space_data_split in tqdm(np.array_split(fair_space_data, fair_space_data.shape[0] // 1_500 + 1)):
                fair_space_data_squared_distances = e_dist_torch(torch.tensor(fair_space_data_split).to(device), torch.tensor(fair_space_data).to(device))
                fair_similarity_W_current = fair_space_data_squared_distances <= tau
                fair_similarity_Ds.append(fair_similarity_W_current.float().sum(1).cpu().numpy())
                fair_similarity_W_current = np.packbits(fair_similarity_W_current.cpu().numpy(), axis=1)
                fair_similarity_Ws.append(fair_similarity_W_current)

            fair_similarity_W = np.concatenate(fair_similarity_Ws, axis=0)
            fair_similarity_D = np.concatenate(fair_similarity_Ds, axis=0)

            y_original = outputs.clone().cpu().numpy()
            y_updated = outputs.clone().cpu().numpy()

            t_e = time.time()
            print('Time (setup) [s]: {:.3f}'.format(t_e - t_s))
            t_s = time.time()

            batch_size = 512

            for iter_idx in trange((y_updated.shape[0] // batch_size) * args.coo_epochs):
                selection = np.random.choice(np.arange(y_updated.shape[0]), batch_size, replace=False)

                fair_similarity_W_selection = fair_similarity_W[selection]
                fair_similarity_W_selection = np.unpackbits(fair_similarity_W_selection, axis=1, count=fair_similarity_W.shape[0]).astype(np.float32)

                fair_similarity_W_selection[np.arange(batch_size), selection] = 0

                avg_degree = fair_similarity_W_selection.sum(1).mean()

                if iter_idx == 0:
                    print('avg_degree', avg_degree.item())

                y_w = (np.expand_dims(y_updated, 0) * np.expand_dims(fair_similarity_W_selection, 2)).sum(1)

                y_updated[selection] = (y_original[selection] + args.lambda_GLIF * y_w) / (1 + args.lambda_GLIF * fair_similarity_W_selection.sum(1, keepdims=True))

            y_updated = torch.tensor(y_updated).to(device)

            t_e = time.time()
            print('Time [s]: {:.3f}'.format(t_e - t_s))

            if args.dataset == 'sentiment':

                names_logits = y_updated[num_test_samples:]
                names_logits = names_logits[:, 1] - names_logits[:, 0]

                eval_model(y_updated[:num_test_samples], names_logits, name='coo_laplacian_tau_{}'.format(tau))

            elif args.dataset == 'bios':

                eval_model(y_updated[:num_test_samples], y_updated[num_test_samples:],
                           name='coo_laplacian_tau_{}'.format(tau))

            elif args.dataset == 'toxicity':
                pred_for_fairness = y_updated[num_test_samples1:]
                pred_for_fairness = pred_for_fairness.reshape(51, X_test_counter.shape[1], 2)
                eval_model(y_updated[:num_test_samples], pred_for_fairness,
                           name='coo_laplacian_tau_{}'.format(tau))

            else:
                raise NotImplementedError(args.dataset)

    ####################################################################################################################
    #  GLIF-NRW: Normalized Random Walk Laplacian Evaluation (coordinate descent)  #####################################
    ####################################################################################################################

    print('')
    print('GLIF-NRW: Normalized Random Walk Laplacian Evaluation (coordinate descent)')
    print('')

    with torch.no_grad():

        for tau in taus:

            print('tau', tau)

            if args.dataset == 'sentiment':
                data = torch.cat([
                    torch.tensor(X_test).float().to(device),
                    torch.tensor(test_names_embed).float().to(device),
                ])
            elif args.dataset == 'bios':
                data = torch.cat([
                    torch.tensor(X_test).float().to(device),
                    torch.tensor(X_counter_test).float().to(device),
                ])
            elif args.dataset == 'toxicity':
                data = torch.cat([
                    torch.tensor(X_test_all).float().to(device),
                ])
            else:
                raise NotImplementedError(args.dataset)

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

            def e_dist_torch(A, B):
                A_n = (A ** 2).sum(dim=1).reshape(-1, 1)
                B_n = (B ** 2).sum(dim=1).reshape(1, -1)
                inner = A @ B.T
                return A_n - 2 * inner + B_n

            fair_similarity_Ws = []
            fair_similarity_Ds = []

            t_s = time.time()

            for fair_space_data_split in tqdm(np.array_split(fair_space_data, fair_space_data.shape[0] // 1_500 + 1)):
                fair_space_data_squared_distances = e_dist_torch(torch.tensor(fair_space_data_split).to(device), torch.tensor(fair_space_data).to(device))
                fair_similarity_W_current = fair_space_data_squared_distances <= tau
                fair_similarity_Ds.append(fair_similarity_W_current.float().sum(1).cpu().numpy())
                fair_similarity_W_current = np.packbits(fair_similarity_W_current.cpu().numpy(), axis=1)
                fair_similarity_Ws.append(fair_similarity_W_current)

            fair_similarity_W = np.concatenate(fair_similarity_Ws, axis=0)
            fair_similarity_D = np.concatenate(fair_similarity_Ds, axis=0)

            D_tildes = []

            for selection in tqdm(np.array_split(np.arange(fair_space_data.shape[0]), fair_space_data.shape[0] // 1_500 + 1)):
                fair_similarity_W_selection = fair_similarity_W[selection]
                fair_similarity_W_selection = np.unpackbits(fair_similarity_W_selection, axis=1, count=fair_similarity_W.shape[0]).astype(np.float32)
                W_tilde = fair_similarity_W_selection / np.sqrt(fair_similarity_D[selection].reshape((-1, 1))) / np.sqrt(fair_similarity_D.reshape((1, -1)))
                D_tildes.append(W_tilde.sum(1))

            D_tilde = np.concatenate(D_tildes, axis=0)

            y_original = outputs.clone().cpu().numpy()
            y_updated = outputs.clone().cpu().numpy()

            t_e = time.time()
            print('Time (setup) [s]: {:.3f}'.format(t_e - t_s))
            t_s = time.time()

            batch_size = 512

            for iter_idx in trange((y_updated.shape[0] // batch_size) * args.coo_epochs):
                selection = np.random.choice(np.arange(y_updated.shape[0]), batch_size, replace=False)

                fair_similarity_W_selection = fair_similarity_W[selection]
                fair_similarity_W_selection = np.unpackbits(fair_similarity_W_selection, axis=1, count=fair_similarity_W.shape[0]).astype(np.float32)

                avg_degree = fair_similarity_W_selection.sum(1).mean()

                if iter_idx == 0:
                    print('avg_degree', avg_degree.item())

                W_tilde = fair_similarity_W_selection / np.sqrt(fair_similarity_D[selection].reshape((-1, 1))) / np.sqrt(fair_similarity_D.reshape((1, -1)))

                one_over_D_tilde_plus_one_over_D_tilde_T = \
                    1 / D_tilde[selection].reshape((-1, 1)) + 1 / D_tilde.reshape((1, -1))

                W_new = W_tilde * one_over_D_tilde_plus_one_over_D_tilde_T / 2

                y_w = np.expand_dims(y_updated, 0) * np.expand_dims(W_new, 2)
                y_w[np.arange(batch_size), selection] = 0
                y_w = y_w.sum(1)

                y_updated[selection] = (y_original[selection] + args.lambda_GLIF_NRW * avg_degree * y_w) / \
                                   (1 + args.lambda_GLIF_NRW * avg_degree *
                                    (1 - np.expand_dims(W_new[np.arange(batch_size), selection], 1)))

            y_updated = torch.tensor(y_updated).to(device)

            t_e = time.time()
            print('Time [s]: {:.3f}'.format(t_e - t_s))

            if args.dataset == 'sentiment':

                names_logits = y_updated[num_test_samples:]
                names_logits = names_logits[:, 1] - names_logits[:, 0]

                eval_model(y_updated[:num_test_samples], names_logits, name='coo_GLIF_NRW_tau_{}'.format(tau))

            elif args.dataset == 'bios':

                eval_model(y_updated[:num_test_samples], y_updated[num_test_samples:],
                           name='coo_GLIF_NRW_tau_{}'.format(tau))

            elif args.dataset == 'toxicity':
                pred_for_fairness = y_updated[num_test_samples1:]
                pred_for_fairness = pred_for_fairness.reshape(51, X_test_counter.shape[1], 2)
                eval_model(y_updated[:num_test_samples], pred_for_fairness,
                           name='coo_GLIF_NRW_tau_{}'.format(tau))

            else:
                raise NotImplementedError(args.dataset)


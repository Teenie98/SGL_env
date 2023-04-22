import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    # for cuda
    parser.add_argument('--cuda', default=-1, type=int, help="Specify CUDA device, defaults to -1 which learns on CPU")
    # for seed
    parser.add_argument('--seed', default=2021, type=int)

    # for gcn
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--layer_num', default=3, type=int)
    parser.add_argument('--eps', default=0.01, type=float)

    # for ssl
    parser.add_argument('--SSL_reg', default=0.1, type=float)
    parser.add_argument('--SSL_dropout_ratio', default=0.1, type=float)
    parser.add_argument('--SSL_temp', default=0.2, type=float)

    # for train
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--stop_cnt', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--reg', default=1e-4, type=float)
    parser.add_argument('--env_mode', default=1, type=int)
    parser.add_argument('--irm_reg', default=0, type=float)

    # for test
    parser.add_argument('--k', default=20, type=int)

    # for save and read
    parser.add_argument('--train_data_path', default='./dataset/yelp2018/yelp2018_with_env.train', type=str)
    parser.add_argument('--test_data_path', default='./dataset/yelp2018/yelp2018_with_env.test', type=str)

    return parser.parse_args()


args = parse_args()

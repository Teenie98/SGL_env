import pandas as pd
import numpy as np
import math


def get_data_path(dataset_name, type, env=False):
    if env:
        path = './{}/{}_with_env.{}'.format(dataset_name, dataset_name, type)
    else:
        path = './{}/{}.{}'.format(dataset_name, dataset_name, type)
    return path


def get_env_dict(cnt_dict, num, K=5):
    env_dic = {}
    rank = sorted(cnt_dict.items(), key=lambda kv: (kv[1], kv[0]))
    for idx, kv in enumerate(rank):
        tag = math.ceil(K * (idx + 1) / num)
        env_dic[kv[0]] = tag
    return env_dic


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv(get_data_path("yelp2018", "train"), sep=',', header=None, names=['user', 'item'])
    test_df = pd.read_csv(get_data_path("yelp2018", "test"), sep=',', header=None, names=['user', 'item'])
    all_df = pd.concat([train_df, test_df])
    # get user/item num
    user_num = len(all_df["user"].unique())
    item_num = len(all_df["item"].unique())

    # count interaction per user/item
    user_count_dict = all_df.groupby("user").count().item.to_dict()
    item_count_dict = all_df.groupby("item").count().user.to_dict()
    # all_df["user_popularity"] = all_df["user"].map(user_count_dict)
    # all_df["item_popularity"] = all_df["item"].map(item_count_dict)

    # add env
    user_env_dict = get_env_dict(user_count_dict, user_num)
    all_df["user_env"] = all_df["user"].map(user_env_dict)
    train_df["user_env"] = train_df["user"].map(user_env_dict)
    test_df["user_env"] = test_df["user"].map(user_env_dict)

    item_env_dict = get_env_dict(item_count_dict, item_num)
    all_df["item_env"] = all_df["item"].map(item_env_dict)
    train_df["item_env"] = train_df["item"].map(item_env_dict)
    test_df["item_env"] = test_df["item"].map(item_env_dict)

    # save data
    train_df.to_csv(get_data_path("yelp2018", "train", 1), index=False, header=True)
    test_df.to_csv(get_data_path("yelp2018", "test", 1), index=False, header=True)

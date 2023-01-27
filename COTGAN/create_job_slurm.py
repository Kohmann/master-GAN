import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COTGAN')
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist | fashion-mnist | cifar10 | cifar100')
    parser.add_argument('--data_path', type=str, default='data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    print(parser.parse_args())

from client_module.log import logger as l
import os
import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms

def find_max_clique(graph):
    '''
    find the max clique of graph,return the n_clique and clique(list of bool,if clique[i] = true means
    the node_i is in the max clique)
    '''
    global n,best_clique,best_clique_n,cur_clique,cur_clique_n
    n = graph.shape[0]
    best_clique = [False for _ in range(n)]
    best_clique_n = 0
    cur_clique = [False for _ in range(n)]
    cur_clique_n = 0
    def can_place(t):
        global cur_clique
        for i in range(t):
            if cur_clique[i] and (not graph[i][t]):
                return False
        return True

    def backward(cur):
        global cur_clique,cur_clique_n,best_clique,best_clique_n
        if(cur >= n):
            # record the best_result
            for i in range(n):
                best_clique[i] = cur_clique[i]
            
            best_clique_n = cur_clique_n
            return 
        # place into cur clique
        if(can_place(cur)):
            cur_clique[cur] = True
            cur_clique_n = cur_clique_n +  1
            backward(cur+1)
            cur_clique_n = cur_clique_n - 1
            cur_clique[cur] = False
        # do not place into cur clique
        if(cur_clique_n + n-1-cur>best_clique_n):
            cur_clique[cur] = False
            backward(cur+1)
    backward(0)
    return best_clique,best_clique_n


def hex2bytes(hex_str):
    '''
    transfer hex_str(type HexStr) to bytes
    '''
    return bytes.fromhex(str(hex_str)[2:])

def build_dataset(dataset_name = 'MNIST',n_client=100,n_attacker=10,data_dir = './data'):
    train_x,train_y,test_x,test_y,n_class= None,None,None,None,None
    if dataset_name == 'MNIST':
        train_x ,train_y,test_x,test_y,n_class = build_MNIST_dataset(data_dir)
    elif dataset_name == 'CIFAR10':
        train_x ,train_y,test_x,test_y,n_class = build_CIFAR10_dataset(data_dir)
    elif dataset_name == 'EMNIST':
        train_x ,train_y,test_x,test_y,n_class = build_EMNIST_dataset(data_dir)
    else:
        raise ValueError('no dataset match')
    return split_dataset_classify(train_x, train_y, test_x, test_y, n_client, n_attacker,n_class)

def build_MNIST_dataset(data_dir = './data'):
    trainset = torchvision.datasets.MNIST(data_dir,train=True,download=True,transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(data_dir,train=False,download=True,transform=transforms.ToTensor())
    train_x = trainset.data.numpy()
    train_y = trainset.targets.numpy()
    test_x = testset.data.numpy()
    test_y = testset.targets.numpy()

    train_x = train_x.reshape(train_x.shape[0],-1)
    test_x = test_x.reshape(test_x.shape[0],-1)
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_x = np.multiply(train_x,1.0/255.0)
    test_x = np.multiply(test_x,1.0/255.0)

    order = np.arange(train_x.shape[0])
    np.random.seed(24)
    np.random.shuffle(order)
    train_x = train_x[order]
    train_y = train_y[order]

    # train_y = dense_to_one_hot(train_y)
    # test_y = dense_to_one_hot(test_y)
    
    l.info(f"build MNIST dataset train_x{train_x.shape} train_y{train_y.shape} test_x{test_x.shape} test_y{test_y.shape}")
    n_class = 10
    return train_x,train_y,test_x,test_y,n_class

def build_EMNIST_dataset(data_dir = './data'):
    trainset = torchvision.datasets.EMNIST(data_dir,train=True,download=True,transform=transforms.ToTensor(),split="letters")
    testset = torchvision.datasets.EMNIST(data_dir,train=False,download=True,transform=transforms.ToTensor(),split="letters")
    train_x = trainset.data.numpy()
    train_y = trainset.targets.numpy()
    test_x = testset.data.numpy()
    test_y = testset.targets.numpy()

    train_x = train_x.reshape(train_x.shape[0],-1)
    test_x = test_x.reshape(test_x.shape[0],-1)
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_x = np.multiply(train_x,1.0/255.0)
    test_x = np.multiply(test_x,1.0/255.0)

    order = np.arange(train_x.shape[0])
    np.random.shuffle(order)
    train_x = train_x[order]
    train_y = train_y[order]

    # train_y = dense_to_one_hot(train_y)
    # test_y = dense_to_one_hot(test_y)
    
    l.info(f"build EMNIST dataset train_x{train_x.shape} train_y{train_y.shape} test_x{test_x.shape} test_y{test_y.shape}")
    n_class = 37
    return train_x,train_y,test_x,test_y,n_class

def build_CIFAR10_dataset(data_dir = './data'):
    image_size = (224,224)
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    trainset=torchvision.datasets.CIFAR10(root=data_dir,train=True,transform=transform,download=True)
    testset=torchvision.datasets.CIFAR10(root=data_dir,train=False,transform=transform,download=True)
    train_x ,train_y = [],[]
    test_x ,test_y = [],[]
    #  when first run it ,change load to false
    load = True
    if load:
        dir = os.path.join(data_dir,"cifar-10-batches-py")
        train_x = np.load(os.path.join(dir,"train_x.npy"))
        train_y = np.load(os.path.join(dir,"train_y.npy"))
        test_x = np.load(os.path.join(dir,"test_x.npy"))
        test_y = np.load(os.path.join(dir,"test_y.npy"))
    else:
        for i in range(trainset.data.shape[0]):
            x,y = (trainset.__getitem__(i))
            if i == 0:
                print(type(x),x.shape,type(y))
            train_x.append(x.numpy())
            train_y.append(y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        for i in range(testset.data.shape[0]):
            x,y = (testset.__getitem__(i))
            if i == 0:
                print(type(x),x.shape,type(y))
            test_x.append(x.numpy())
            test_y.append(y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        dir = os.path.join(data_dir,"cifar-10-batches-py")
        np.save(os.path.join(dir,"train_x.npy"),train_x)
        np.save(os.path.join(dir,"train_y.npy"),train_y)
        np.save(os.path.join(dir,"test_x.npy"),test_x)
        np.save(os.path.join(dir,"test_y.npy"),test_y)

    l.info(f"build CIFAR10 dataset train_x{train_x.shape} train_y{train_y.shape} test_x{test_x.shape} test_y{test_y.shape}")
    order = np.arange(train_x.shape[0])
    # shuffle
    np.random.seed(24)
    np.random.shuffle(order)
    train_x = train_x[order]
    train_y = train_y[order]
    
    n_class = 10
    return train_x,train_y,test_x,test_y,n_class

    
def split_dataset_classify(train_x,train_y,test_x,test_y,n_client,n_attacker,n_class):
    train_x_size = train_x.shape[0]
    shard_size = train_x_size// n_client // 2
    # the id list of shard, every shard's size is [shard_size]
    np.random.seed(24+125)
    shard_ids= np.random.permutation(train_x_size // shard_size)
    l.info(f"train_x_size:{train_x_size},shard_size:{shard_size},n_shard:{len(shard_ids)}")
    split_dataset_tensor = {}
    
    attackers = np.arange(n_attacker)
    l.info(f'exist attacker as follow:{attackers}')
    random_labels = np.random.permutation(n_class)
    for i in range(n_client):
        first_id = shard_ids[i * 2]
        second_id = shard_ids[i * 2 + 1]
        # if i = 10 ,first_id = 10 * 2 = 20 ,second_id = 10 * 2 + 1 =20
        first_shard = train_x[first_id * shard_size: first_id * shard_size + shard_size]
        second_shard = train_x[second_id * shard_size: second_id * shard_size + shard_size]
        label_shards1 = train_y[first_id * shard_size: first_id * shard_size + shard_size]
        label_shards2 = train_y[second_id * shard_size: second_id * shard_size + shard_size]
        client_train_x, client_train_y = np.vstack((first_shard, second_shard)), np.hstack([label_shards1, label_shards2])
        if i==0:
            l.info(f"client_train_x.shape:{client_train_x.shape},client_train_y.shape:{client_train_y.shape}")
        if i in attackers:
            
            l.debug(f"for client{i}, before random,labers is {client_train_y[:10]},random is{ random_labels}")

            client_train_y = np.array(list(map(lambda x:random_labels[x],client_train_y)))
            l.debug(f"after random,labers is {client_train_y[:10]}")
            
        x_tensor ,y_tensor= torch.tensor(client_train_x), torch.tensor(client_train_y)
        if i == 0:
            l.info(f"splited_train_x_tensor.shape:{x_tensor.shape},splited_train_y_tensor.shape:{y_tensor.shape}")
        split_dataset_tensor[i]=(x_tensor,y_tensor)

    test_x_tensor = torch.tensor(test_x)
    test_y_tensor = torch.tensor(test_y)
    return split_dataset_tensor,test_x_tensor,test_y_tensor




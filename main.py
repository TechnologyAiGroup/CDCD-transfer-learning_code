import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import argparse
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
# from test import test
from utils import calacc, calDVR

# source_dataset_name = 'MNIST'
# target_dataset_name = 'mnist_m'
# source_image_root = os.path.join('dataset', source_dataset_name)
# target_image_root = os.path.join('dataset', target_dataset_name)


def run(args):
    source_dataset_name = args.s
    target_dataset_name = args.t
    metrictype = "a"
    source_image_root = os.path.join('dataset', source_dataset_name+"_train")
    target_image_root = os.path.join('dataset', target_dataset_name+"_train")
    print(source_dataset_name, target_dataset_name)

    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 0.01
    batch_size = 128
    # image_size = 28
    image_size = (60, 60)
    n_epoch = 50
    print(f"lr: {lr}, batch size: {batch_size}")

    manual_seed = random.randint(1, 10000)
    print(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # dataset_source = datasets.MNIST(
    #     root='dataset',
    #     train=True,
    #     transform=img_transform_source,
    #     download=True
    # )

    dataset_source = GetLoader(
        # data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_root=source_image_root,
        data_list=os.path.join("dataset/", f'{source_dataset_name}_{metrictype}_train.txt'),
        transform=img_transform_source
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    # train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    # train_list = os.path.join("dataset/", f'{target_dataset_name}_{metrictype}_train.txt')

    dataset_target = GetLoader(
        # data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_root=target_image_root,
        # data_list=train_list,
        data_list=os.path.join("dataset/", f'{target_dataset_name}_{metrictype}_train.txt'),
        transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    # load model

    my_net = CNNModel()

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    secondary_accu_t = 0.0
    list_epoch_accs_acct_dvrs_dvrt = []
    premodel = None
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label = domain_label.cuda()


            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            #sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
            #                % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
            #                    err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            # sys.stdout.flush()
            torch.save(my_net, '{0}/{1}_{2}_model_epoch_current.pth'.format(model_root, source_dataset_name, target_dataset_name))

        print(f'epoch: {epoch}*****************************************\n')
        # accu_s = test(dataset_name=source_dataset_name, resize=image_size, metrictype=metrictype, flag="s")
        # print('Accuracy of the %s dataset: %f' % (source_dataset_name, accu_s))
        # accu_t = test(dataset_name=target_dataset_name, resize=image_size, metrictype=metrictype, flag="t")
        # print('Accuracy of the %s dataset: %f' % (target_dataset_name, accu_t))
        
        # accu_s, resp_s = calacc(resize=image_size, metrictype=metrictype, flag="s", source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name)
        # dvr_s = calDVR(testset_root=source_dataset_name, id_resp=resp_s)
        # print('Accuracy of the %s dataset: %f' % (source_dataset_name, accu_s))
        # print('DVR of the %s dataset: %f %%' % (source_dataset_name, dvr_s*100))
        accu_t, resp_t = calacc(resize=image_size, metrictype=metrictype, flag="t", source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name)
        dvr_t = calDVR(testset_root=target_dataset_name, id_resp=resp_t)
        print('Accuracy of the %s dataset: %f' % (target_dataset_name, accu_t))
        print('DVR of the %s dataset: %f %%' % (target_dataset_name, dvr_t*100))
        
        # list_epoch_accs_acct_dvrs_dvrt.append([epoch, accu_s, accu_t, dvr_s, dvr_t])
        list_epoch_accs_acct_dvrs_dvrt.append([epoch, accu_t, dvr_t])
        
        if accu_t > best_accu_t:
            # best_accu_s = accu_s
            secondary_accu_t = best_accu_t
            best_accu_t = accu_t
            if not premodel:
                premodel = my_net
            else:
                torch.save(premodel, '{0}/{1}_{2}_model_epoch_secondary.pth'.format(model_root, source_dataset_name, target_dataset_name))
                premodel = my_net
            torch.save(my_net, '{0}/{1}_{2}_model_epoch_best.pth'.format(model_root, source_dataset_name, target_dataset_name))

    print('============ Summary ============= \n')
    # print('Accuracy of the %s dataset: %f' % (source_dataset_name, best_accu_s))
    print('Accuracy of the %s dataset(best): %f' % (target_dataset_name, best_accu_t))
    print('Accuracy of the %s dataset(secondary): %f' % (target_dataset_name, secondary_accu_t))
    import pandas as pd
    csvfile = pd.DataFrame(list_epoch_accs_acct_dvrs_dvrt)
    csvfile.columns = ["epoch", f"{target_dataset_name} acc", f"{target_dataset_name} dvr"]
    savefold = "./data_0228"
    if not os.path.exists(savefold):
        os.makedirs(savefold)
    csvfile.to_csv(os.path.join(savefold, f"{source_dataset_name}-{target_dataset_name}.csv"), index=False)
    # print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input source and target")
    parser.add_argument("-s", type=str, default="ctrl", help="source")
    parser.add_argument("-t", type=str, default="ctrl", help="target")
    args = parser.parse_args()
    print(args)
    run(args)
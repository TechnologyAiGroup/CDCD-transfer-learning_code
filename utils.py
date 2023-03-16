# add on 20230216
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
import re
from PIL import Image
import numpy as np


def calacc(resize, metrictype, flag, source_dataset_name, target_dataset_name):
    """ test """
    # Model
    cuda = True
    alpha = 0
    model_root = "models"
    my_net = torch.load('{0}/{1}_{2}_model_epoch_current.pth'.format(model_root, source_dataset_name, target_dataset_name))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    # Dataset     
    # eg: testset_root "ctrl"
    if flag=="s":
        testset_root = "./dataset/"+source_dataset_name
    elif flag=="t":
        testset_root = "./dataset/"+target_dataset_name

    testresp = testset_root+"_test_list.txt"
    with open(testresp, "r") as f:
        testset = f.readlines()
    f.close()

    with open(testset_root+f"_{metrictype}_test.txt", "r") as f:
        nameys = f.readlines()
    f.close()
    # print(nameys)

    img_transform_source = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    totalchip = 0
    correct = 0
    incorrect = 0
    id_resp = [-1]*len(testset)
    for p in range(len(testset)):
        [chip, fault, respinfo] = testset[p].split("/")[1:4]
        resp = []
        for namey in nameys:
            s = f"{fault}-{respinfo}"[:-1]          # delete '\n'
            if re.search(s, namey):
                resp.append(namey)
        resp.sort()
        if len(resp)<=0:
            continue
        totalchip+=1
        
        
        # 处理一组resp，对应一个label 0或1
        temproot = "temp"
        if not os.path.exists(temproot):
            os.makedirs(temproot)
        if flag == "s":
            with open(f"{temproot}/temp_s_{source_dataset_name}-{target_dataset_name}.txt", "w") as f:
                f.truncate(0)
                for item in resp:
                    f.write(item)
            f.close()
            dataset = GetLoader(
                data_root=testset_root+"_test",
                # data_list=testset_image_root.split("_")[0]+f"_{metrictype}_test.txt",
                data_list=f"{temproot}/temp_s_{source_dataset_name}-{target_dataset_name}.txt",
                transform=img_transform_source
            )
        elif flag=="t":
            with open(f"{temproot}/temp_t_{source_dataset_name}-{target_dataset_name}.txt", "w") as f:
                f.truncate(0)
                for item in resp:
                    f.write(item)
            f.close()
            dataset = GetLoader(
                data_root=testset_root+"_test",
                # data_list=testset_image_root.split("_")[0]+f"_{metrictype}_test.txt",
                data_list=f"{temproot}/temp_t_{source_dataset_name}-{target_dataset_name}.txt",
                transform=img_transform_target
            )

        batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

        data_target_iter = iter(dataloader)
        
        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target
        # print(t_img, t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=False)[1]
        assert len(pred)==len(t_label)
        guard = 2
        isStop = False
        if pred.sum()==0:
            correct+=1
            continue
        # for i in range(len(pred)):
        #     if pred[i]==0:
        #         continue
        #     elif pred[i]==1:
        #         id_resp[p] = i
        #         if t_label[i]==1:
        #             correct += 1
        #             break
        #         else:
        #             incorrect += 1
        #             break
        #     else:
        #         raise Exception
        #         exit()
        
        for i in range(guard-1, len(pred)):
            if pred[i]==0:
                continue
            elif pred[i]==1:
                if sum(pred[i-guard:i+1])==1:
                    # 满足保护带机制
                    id_resp[p] = i
                    isStop = True
                    if t_label[i]==1:
                        correct+=1
                        break
                    else:
                        incorrect+=1
                        break
                else:
                    # 不满足保护带机制
                    continue
            else:
                raise Exception
        if not isStop:
            correct+=1
            continue

    if correct+incorrect==totalchip:
        return float(correct/totalchip), id_resp
    else:
        raise Exception


def calDVR(testset_root, id_resp):
    # Dataset
    testresp = "./dataset/"+testset_root+"_test_list.txt"
    with open(testresp, "r") as f:
        testset = f.readlines()
    f.close()

    assert len(testset)==len(id_resp)
    dvr = 0.0
    for i in range(len(testset)):
        [chip, fault, respinfo] = testset[i].split("/")[1:4]
        respinfo = respinfo[:-1]    # 因为存入文件时多存入了一个换行符
        if id_resp[i]==-1:
            continue
        # failpath = os.path.join("./pic/", chip, fault, respinfo, f"{fault}-{respinfo}_{id_resp[i]+1}.bmp")
        # allfailpath = os.path.join("./pic/", chip, fault, respinfo, f"{fault}-{respinfo}_all.bmp")
        failpath = os.path.join("./dataset/", f"{chip}_test", f"{fault}-{respinfo}_{id_resp[i]+1}.bmp")
        allfailpath = os.path.join("./dataset/", f"{chip}_test", f"{fault}-{respinfo}_all.bmp")
        if not os.path.exists(failpath):
            failpath = allfailpath
        imgi = np.asarray(Image.open(failpath))
        if os.path.exists(allfailpath):
            imgall = np.asarray(Image.open(allfailpath))
        else:
            # back = os.listdir(os.path.join("./pic/", chip, fault, respinfo))
            # imgall = np.asarray(Image.open(os.path.join("./pic/", chip, fault, respinfo, f"{fault}-{respinfo}_{len(back)-1}.bmp")))
            back = [j for j in os.listdir(os.path.join("./dataset/", f"{chip}_test")) if j.startswith(f"{fault}-{respinfo}")]
            # back.sort()
            # print(back)
            imgall = np.asarray(Image.open(os.path.join("./dataset/", f"{chip}_test", f"{fault}-{respinfo}_{len(back)}.bmp")))

        fenzi = np.sum([imgi==255])
        fenmu = np.sum([imgall==255])
        assert fenzi<=fenmu
        dvr = dvr + (1-(fenzi/fenmu))

    return dvr/len(testset)

# r, id_resp = calacc("./dataset1/ctrl_test", (28,28),"b")
# s = calDVR("./dataset1/ctrl_test", id_resp)
# print(r, s)
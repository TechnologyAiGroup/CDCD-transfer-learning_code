import os
import shutil
import random
import argparse
# from my_utils import splitDataset


def build(dataset, forwhat, type="b"):
    # picroot = "./pic/" + chip + "/" + fault + "/"
    assert forwhat in ["train", "test"]
    root = "./dataset"
    if not os.path.exists(root):
        os.makedirs(root)
    chip = dataset[0].split("/")[1]
    picroot = f"{root}/{chip}_{forwhat}"
    
    with open(os.path.join(root, f"{chip}_{type}_{forwhat}.txt"), "w") as f:
        f.truncate(0)
    f.close()

    if not os.path.exists(picroot):
        os.makedirs(picroot)
    else:
        shutil.rmtree(picroot)
        # os.system(f"rm -r {picroot}")
        print(f"已清空 path:{picroot}")
        os.makedirs(picroot)
    count = 0  # 采样限制

    for resproot in dataset:
        # [fault, respinfo] = resproot.split("/")[2:4]
        labelfile = "labels.txt"
        if type == "a":
            # labelfile = "labels_ma.txt"
            labelfile = "labels_modified.txt"
        
        if os.path.exists(os.path.join(resproot, labelfile)):
            # 如果该文件夹有效
            for item in os.listdir(resproot):
                if "bmp" in item:
                    sourcename = os.path.join(resproot, item)
                    targetname = os.path.join(picroot, item)
                    shutil.copyfile(sourcename, targetname)
                elif "labels_ma.txt"==item:
                    with open(os.path.join(resproot, labelfile), "r") as sf:
                        srclines = sf.readlines()
                    sf.close()
                    with open(os.path.join(root, f"{chip}_{type}_{forwhat}.txt"), "a") as tf:
                        for l in srclines:
                            tf.write(l)
                    tf.close()


def splitDataset(circuit, fault_types):
    # 按比例划分数据集
    allchips = []
    for fault in fault_types:
        root = os.path.join("pic/", circuit, fault)
        if not os.path.exists(root):
            continue
        perchip =os.listdir(root)
        for chip in perchip:
            allchips.append(os.path.join(root, chip))
    random.seed()
    random.shuffle(allchips)
    index = int(0.9*len(allchips))
    trainset = allchips[:index]
    testset = allchips[index:]

    # print(len(trainset))
    # print(len(testset))
    # print(testset)
    return trainset, testset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get label mb")
    parser.add_argument("-c", type=str, default=["ctrl"], nargs="+", help="circuits/chips")
    parser.add_argument("--type", type=str, default=["and", "or", "fe", "dom", "ssl", "msl"], nargs="+", help="fault types")
    args = parser.parse_args()
    print(args)
# chips = ["int2float", "dec", "priority", "sin", "ctrl", "cavlc", "i2c", "adder", "bar"]
# chips = ["x1", "pair", "frg2", "i10", "des"]
# chips = ["b15"]
# faults = ["ssl", "msl", "and", "or", "fe", "dom"]
    chips = args.c
    faults = args.type

    for chip in chips:
        root = "./dataset"
        train, test = splitDataset(circuit=chip, fault_types=faults)
        print(f"已划分{chip}")
        # print(train)
        # exit()
        build(dataset=train, forwhat="train", type="a")
        build(dataset=test, forwhat="test", type="a")
        print(f"已生成{chip}，正写入。")
        cmd = f"scp -r ./dataset/{chip}_* pc09@222.20.126.25:/home/pc09/user/xhc/experiments/DANN/sourceCode_DANN_py3/dataset1/"
        
        
        with open(f"{root}/{chip}_train_list.txt", "w") as f:
            for i in train:
                f.write(i+"\n")
        f.close()
        with open(f"{root}/{chip}_test_list.txt", "w") as f:
            for i in test:
                f.write(i+"\n")
        f.close()

        # os.system(cmd)
        print(f" {chip} done ")
        
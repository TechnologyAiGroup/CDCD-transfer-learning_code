import os
import argparse


def increaseandthreshold(xys, threshold):
    xs = []
    ys = []
    biggest = 0
    for xy in xys:
        [x, y] = xy.split(" ")
        y = float(y)
        xs.append(x)
        if y<=biggest:
            ys.append(biggest)
        else:
            ys.append(y)
            biggest = y

    for k in range(len(ys)):
        if ys[k]<=threshold:
            ys[k]=0
        else:
            ys[k]=1
            
    assert len(xs)==len(ys)
    return [f"{xs[i]} {ys[i]}\n" for i in range(len(xs))]


root = "./pic"
labelfile = "labels_ma.txt"

# chips = os.listdir(root)
# chips = ["int2float", "dec", "priority", "adder"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="modify labels")
    parser.add_argument("-c", type=str, default=["ctrl"], nargs="+", help="circuits/chips")
    parser.add_argument("--type", type=str, default=["and", "or", "fe", "dom", "ssl", "msl"], nargs="+", help="fault types")
    parser.add_argument("-t", type=float, default=0.899999, help="ma threshold")
    args = parser.parse_args()
    print(args)
    # chips = ["b12", "b14", "b15"]
    # faults = ["ssl", "msl", "and", "or", "fe", "dom"]
    # threshold = 0.899999
    chips = args.c
    faults = args.type
    threshold = args.t
    for chip in chips:
        for fault in faults:
            if not os.path.exists(os.path.join(root, chip, fault)):
                continue
            resps = os.listdir(os.path.join(root, chip, fault))
            for resp in resps:
                item = os.path.join(root, chip, fault, resp, labelfile)
                if not os.path.exists(item):
                    print(item)
                    raise Exception
                with open(item, "r") as f:
                    xys = f.readlines()
                f.close()
                xys = increaseandthreshold(xys=xys, threshold=threshold)
                with open(os.path.join(root, chip, fault, resp, "labels_modified.txt"), "w") as f:
                    for i in xys:
                        f.write(i)
                f.close()
        print(f"{chip} done")

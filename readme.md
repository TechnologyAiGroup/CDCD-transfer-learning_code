In this example, the chip name is fixed as "x1" and "pair".  
We have generated datasets named "x1" and "pair".  

## Generate datasets:
> `python my_utils.py -c x1 pair (--type ssl msl and or dom fe)`  
> Generate `./pic` folder and generate pictures

> `python getlabel_ma.py -c x1 pair (--type ssl msl and or dom fe)`   
> Generate `lables_ma.txt` in './pic/chip(X)/fault(Y)/(Z)_resp/'  

> `python modify_label.py -c x1 pair (--type ssl msl and or dom fe) (-t 0.899999)`  
> Modify the labels into 0/1  

> `python buildDataset.py -c x1 pair (--type ssl msl and or dom fe)`  
> Build the dataset from the pic folder and divide it by 9:1 (trainset : testset)

> 
> -c: chip name  
> --type: fault type (default:["and", "or", "fe", "dom", "ssl", "msl"])  
> -t: threshold for modifying ma labels  

----
## Decision Tree:
> `python DAC4.py -s x1 -t pair`  
> Decision Tree, trained with chip x1's trainset, tested with chip pair's testset

> 
> -s: source data  
> -t: target data


## CDCD with DANN:  
> `python main.py -s x1 -t pair`  
> DANN, trained with chip x1's trainset with labels and chip pair's trainsset without labels, tested with chip pair's testset
>   
> -s: source data  
> -t: target data

In this example, the chip name is fixed as "x1".  
We have generated datasets named "x1" and "pair".  

## Generate datasets:
> python my_utils.py -c x1 pair (--type ssl msl and or dom fe)  
> python getlabel_ma.py -c x1 pair (--type ssl msl and or dom fe)  
> python modify_label.py -c x1 pair (--type ssl msl and or dom fe) (-t 0.899999)  
> python buildDataset.py -c x1 pair (--type ssl msl and or dom fe)  
> 
> -c: chip name  
> -type: fault type (default:["and", "or", "fe", "dom", "ssl", "msl"])  
> -t: threshold for modifying ma labels  


## Decision Tree:
> python DAC4.py -s x1 -t pair  
> 
> -s: source data  
> -t: target data


## CDCD with DANN:  
> python main.py -s x1 -t pair  
>   
> -s: source data  
> -t: target data

# CB-SCTC
The pytorch code for our paper 'Cross-block Sparse Class Token Contrast  for Weakly Supervised Semantic Segmentation'.

[[Project Page]](https://xulianuwa.github.io/MCTformer-project-page/)

<p align="center">
  <img src="cb-sctc_6.png" width="1080" title="Overview of CB-SCTC" >
</p>
<p align = "center">
Fig.1 - Overview of CB-SCTC
</p>

## Prerequisite
- Ubuntu 20.04, with Python 3.7 and the following python dependencies.
```
pip install -r requirements.txt
```
- Download [the PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).

## Run for PASCAL VOC 2012 dataset
```
bash run_for_pascalVoc2012.sh
```
## Run for MS COCO 2014 dataset
```
bash run_for_msCoco2014.sh
```

## Contact
If you have any questions, you can either create issues or contact me by email
[tangjingfeng@stmail.ujs.edu.cn](tangjingfeng@stmail.ujs.edu.cn)

## Acknowledgement
We heavily borrowed [MCTFormerV1](https://github.com/xulianuwa/MCTformer) to construct our backbone. Many thanks to their brilliant works!

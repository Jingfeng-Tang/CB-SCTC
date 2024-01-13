# CB-SCTC
The pytorch code for our paper ''.
test
[[Project Page]](https://github.com/Jingfeng-Tang/CB-SCTC)

<p align="center">
  <img src="fig2_cb-sctc_8.png" width="1080" title="The architecture of the proposed WSSS framework(CB-SCTC)" >
</p>
<p align = "center">
Fig.1 - The architecture of the proposed WSSS framework(CB-SCTC)
</p>

## Prerequisite
- Ubuntu 20.04, with Python 3.7 and the following python dependencies.
```
pip install -r requirements.txt
```
- Download [the PASCAL VOC 2012 training and validation sets](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
- Download [the PASCAL VOC 2012 test set](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar).
- Download [the PASCAL VOC 2012 SBD set](https://drive.google.com/file/d/1doCUI9h_lxhxIS7WZX8SSXpDyCjIEwZj/view?usp=drive_link).

- Download [the MS COCO 2014 training set](http://images.cocodataset.org/zips/train2014.zip).
- Download [the MS COCO 2014 validation set](http://images.cocodataset.org/zips/val2014.zip).
- Download [the MS COCO 2014 training and validation annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).

- Download [the segNet pre-trained weight](https://drive.google.com/file/d/1TKYhIq1uxnEgv6oX9Exc09uPJ4bf95eQ/view?usp=drive_link).


## Run for PASCAL VOC 2012 dataset
```
bash run_for_PASCALVOC2012.sh
```
## Run for MS COCO 2014 dataset
```
bash run_for_MSCOCO2014.sh
```

## Contact
If you have any questions, you can either create issues or contact me by email
[tangjingfeng@stmail.ujs.edu.cn](tangjingfeng@stmail.ujs.edu.cn)

## Acknowledgement
We heavily borrowed [MCTFormerV1](https://github.com/xulianuwa/MCTformer) to construct our backbone. Many thanks for their brilliant work!

#!/bin/bash

# Semantic segmentation datasets
wget "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
wget "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
# Note: Mapillary requires a sign-up and cannot be directly downloaded via wget
# Note: PACO-LVIS and PASCAL-Part are hosted on GitHub and might require a different method for downloading
wget "http://images.cocodataset.org/zips/train2017.zip"

# Referring segmentation datasets
wget "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
wget "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"
wget "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"
wget "https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip"
wget "https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip"
# Note: OneDrive link requires a different method for downloading

# Visual Question Answering dataset
# Note: LLaVA-Instruct-150k is hosted on Hugging Face and requires the datasets library for downloading
git lfs clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K.git dataset/LLaVA-Instruct-150k

# Reasoning segmentation dataset
# Note: ReasonSeg is hosted on GitHub and might require a different method for downloading

# Unzipping and arranging datasets
unzip ADEChallengeData2016.zip -d dataset/ADE20K
unzip stuffthingmaps_trainval2017.zip -d dataset/COCO-Stuff
unzip train2017.zip -d dataset/coco/train2017
unzip refcoco.zip -d dataset/refCOCO
unzip refcoco+.zip -d dataset/refCOCO+
unzip refcocog.zip -d dataset/refCOCOg
unzip refclef.zip -d dataset/refCLEF
unzip saiapr_tc-12.zip -d dataset/saiapr_tc-12

echo "Datasets downloaded, unzipped, and arranged. Some datasets require manual download or additional methods."
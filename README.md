# CLActive

This is a fork of the repository of ["Selection via Proxy: Efficient Data Selection for Deep Learning"](https://openreview.net/forum?id=HJg2b0VYDr) with a few modifications. 


If you use this code in your research, please use the following BibTeX entry of the SVP paper.

```
@inproceedings{
    coleman2020selection,
    title={Selection via Proxy: Efficient Data Selection for Deep Learning},
    author={Cody Coleman and Christopher Yeh and Stephen Mussmann and Baharan Mirzasoleiman and Peter Bailis and Percy Liang and Jure Leskovec and Matei Zaharia},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=HJg2b0VYDr}
}
```

##### Setup

Refer to the original SVP repository for setup and general usage examples. 

##### Examples

```bash
# Perform active learning on CIFAR-10 with a Memory Ratio of 1
python3.7 -m svp.cifar active --dataset cifar10 --arch preact164 --num-workers 4 \
      --selection-method least_confidence \
      --proxy-arch preact20 \
      --proxy-learning-rate 0.01 --proxy-epochs 1 \
      --proxy-learning-rate 0.1 --proxy-epochs  $i \
      --proxy-learning-rate 0.01 --proxy-epochs 4 \
      --initial-subset 1000 \
      --round 4000 \
      --round 5000 --round 5000 \
      --round 5000 --round 5000 \
      --device 0 --run-dir "$run_dir" \
      --memory-ratio 1
```

```bash
# Perform active learning on Amazon Reviews Polarity with fastText.
python3.7 -m svp.amazon fasttext '../fastText-0.9.2/fasttext' --run-dir . --datasets-dir '../Amazon' --dataset amazon_review_polarity --selection-method least_confidence --size 72000 --size 360000 --size 720000 --size 1080000 --size 1440000 --size 1800000

# Use selected labeled data from fastText to train VDCNN29
python3.7 -m svp.amazon active --run-dir . --datasets-dir '../Amazon' --dataset amazon_review_polarity --num-workers 8 --arch vdcnn29-conv --selection-method least_confidence --precomputed-selection $path_to_fasttext_selections --eval-target-at 360000 --eval-target-at 720000 --eval-target-at 1080000 --eval-target-at 1440000 --eval-target-at 1800000

```




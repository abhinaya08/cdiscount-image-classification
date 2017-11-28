# cdiscount-image-classification
All codes for cdiscount image [classification](https://www.kaggle.com/c/cdiscount-image-classification-challenge).

Data visualization and analysis: [site](https://www.kaggle.com/vfdev5/data-visualization-and-analysis)

### How to train
```
python main.py
```

### How each file function
| file name       | notes                                    |
| :-------------- | :--------------------------------------- |
| data transform  | contain various data augmentation functions |
| label id dict   | two dictions, mapping from category id to category label (a integer between 1 and 5270) and from category label to category id respectively |
| se inception v3 | network                                  |
| Trainer         | The first trainer ensembling all parts together to train the model, but suffers from poor flexibility and need to be improved |
| train se inv3   | The improved trainer, but still exist some bugs |
| utils           | contain important useful functions, including data loading and learning rate adjustment |


### To be implemented
- [ ] train another model  (resnet 101 or resnet152, but the later seems to work better)  for results ensembling.
- [x] split a fraction (about one tenth) of training data for fast experiments
- [x] maybe we can improve the efficiency of data loading by writing the location and length of all training images into a csv file, so that we can avoid to loading all data information at every beginning of training. 
- [x] add some augmentation functions.
- [ ] retrain hard samples.
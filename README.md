# Deep-Image-Ranking

Neural Networks have been used for a variety of tasks, especially using unstructured data. Neural Networks are extremely good at image recognition, image segmentation etc. Learning Fine-grained Image Similarity with Deep Ranking (https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf) is a novel application of neural networks, where the authors use a new multi scale architecture combined with a triplet loss to create a neural network that is able to perform image search.

This repository is a simpler implementation of the paper. The differences is that, the entire multi scale network has been replaced by a resnet. A simpler version of triplet sampling has been used. 

### Specifics :
Network Used : Resnet 50

Dataset Used for training the network : tiny-image-net (http://cs231n.stanford.edu/tiny-imagenet-200.zip)

Trained on : K20 nvdia

Epochs : 11

Total training time : 20 Hours

### Sample output
Sample results from the network are as shown below :

Query Image : 

![im1](https://github.com/SathwikTejaswi/Deep-Image-Ranking/blob/master/sample_outputs/example1/query.JPEG)

Results :

![im2](https://github.com/SathwikTejaswi/Deep-Image-Ranking/blob/master/sample_outputs/example1/result1.JPEG)
![im2](https://github.com/SathwikTejaswi/Deep-Image-Ranking/blob/master/sample_outputs/example1/result2.JPEG)
![im2](https://github.com/SathwikTejaswi/Deep-Image-Ranking/blob/master/sample_outputs/example1/result3.JPEG)
![im2](https://github.com/SathwikTejaswi/Deep-Image-Ranking/blob/master/sample_outputs/example1/result4.JPEG)
![im2](https://github.com/SathwikTejaswi/Deep-Image-Ranking/blob/master/sample_outputs/example1/result5.JPEG)

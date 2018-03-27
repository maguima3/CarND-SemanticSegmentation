# Semantic Segmentation

### Introduction
The goeal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN).
This meas that each pixel is going to be classified as "road" or as "not road".

The paper "Fully Convolutional Networks for Semantic Segmentation" from UC Berkeley explains the methodology and architecture I followed in this project in detail.
The code can be found [here](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/net.py).

The FCN can be divided in two parts: the **encoder** and the **decoder**.
* Encoder.
The encoder extracts features from the images, which will be used by the decoder.
As suggested in the paper and in the lessons, I have used a VGG16 model pre-trained on ImageNet.
To perserve the spatial information, the fully-connented layers of the VGG16 model have been replaced
with convolutions.
* Decoder.
Using transposed convolutional and skip layers, the decoder helps mapping
low resolution feature maps at the output of the encoder to full input image size feature maps.

### Final outcome
The FCN was trained with the following parameters:
* Epochs: 90
* Batch size: 10
* Learning rate: 1e-4
* Keep probability: 0.5

During training, I have checked that the loss of the model decreased over time. The final average loss was
````
EPOCH 90 ...
Loss: = 0.076
Loss: = 0.083
Loss: = 0.078
Loss: = 0.088
Loss: = 0.081
Loss: = 0.085
Loss: = 0.081
Loss: = 0.080
Loss: = 0.090
Loss: = 0.089
Loss: = 0.079
Loss: = 0.088
Loss: = 0.083
Loss: = 0.086
Loss: = 0.093
Loss: = 0.083
Loss: = 0.091
Loss: = 0.085
Loss: = 0.081
Loss: = 0.087
Loss: = 0.084
Loss: = 0.091
Loss: = 0.084
Loss: = 0.084
Loss: = 0.086
Loss: = 0.084
Loss: = 0.082
Loss: = 0.094
Loss: = 0.079
Average epoch loss: = 0.085
````

The results obtained labels at least 80% of the road pixels correctly, and labels no more than 20% of the non-road pixels as road.

The images bellow shows the performance of the FCN on the ten first images of the test-set.

![](./result_images/um_000000.png)

![](./result_images/um_000001.png)

![](./result_images/um_000002.png)

![](./result_images/um_000003.png)

![](./result_images/um_000004.png)

![](./result_images/um_000005.png)

![](./result_images/um_000006.png)

![](./result_images/um_000007.png)

![](./result_images/um_000008.png)

![](./result_images/um_000009.png)

![](./result_images/um_000010.png)





### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.

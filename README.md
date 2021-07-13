# Keypoints Tracking via Transformer Networks

Model for sparse keypoints tracking across images using transformer networks

Our approach is hierarchical since a coarse keypoint tracking is accurately refined by a second transformer network. The model can be be used for both: image matching, and keypoint tracking 

## Architecture

<img src="./media/arc2.png" width="640" height="200">

## Demo

### Image matching 

Match two image using [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) descriptors 
(Basically tracking keypoints extracted by SuperPoint  ) 

- python demo_2_im.py --image1_path=path1 --image2_path=path2
- python demo_2_im.py                                          if you want to save the result\

For example : 

python demo_2_im.py --image1_path ="./media/im1.jpg" --image2_path="./media/im2.jpg"

![alt text](./results/res.jpg)


 ### Point Tracking

Tracking the points specified in demo_point_tracking.py

 
- python demo_point_tracking.py --image1_path=path1 --image2_path=path2

 ![alt text](./results/res_track.jpg)
 

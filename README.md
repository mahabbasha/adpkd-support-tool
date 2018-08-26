# ADPKD Support Tool

### Important notices

This software is still under development. You can use it, but you're not safe from bugs. If you find some: Open an issue. The software is still in german, so use an dictionary or translate it on yourself!

### What is does for your research

In medical research of PCKD it is useful to mark and calculate the area and volume of harmful cells. With on going development of object detection and instance segmentation with neural networks, this tool uses a trained Mask RCNN model for this task. After identifying possible harmful cysts in your image you can delete, resize created annotations from the network or create new annotations in addition. On top of that you can also modify and delete the created masks and create new masks for each detected/marked cell. After you finished, you can create images (from the original ones) with segmentation masks and area calculations in it. There is also an option to create a Excel Sheet as overview of all marked cells and information about them.

### Packages and usage

`Python>=3.5`

`tensoflow>=1.5.0`

`keras==2.0.8`

`OpenCV >= 3.4.0`

`skimage>=0.14dev`

`PyQt5`

`numpy`

`pillow>=5.2.0`

`shapely>=1.6.4.post2`

If i forgot something to mention, install it on yourself and leave a issue for me! 

Once you installed all needed packages, you can just simply start the tool by entering the folder and type `python3 main.py`. 

If you don't have a trained Mask RCNN network for object detection and instance segementation, you can train it on your own dataset. How to train it, is described [here](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). 

In the future i may publish my trained network. 

###  Acknowledgments

GUI based on [LabelImg](https://github.com/tzutalin/labelImg)

Mask RCNN Implemenation from [Mask RCNN (Matterport)](https://github.com/matterport/Mask_RCNN)

### License

See License. It's MIT.

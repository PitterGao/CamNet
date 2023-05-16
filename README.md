# <center>GeoCamNet: An Extremely Light Carbonate Classification Model Distilled from An Extremely Deep ResNet101 Model 

<div align=center><img src=".\doc\img\heatmap.gif" width="300px" loc="center"/></div>
 
**<center>Fig 1. GIF performance of the model</center>**

## Introduction
Our main tasks so far are to train a microscopic carbonate images classification model on a extremely large image datasets (<b>***Fig. 1***</b>). The datasets include over 10K+ high-resolution optical images. The images are mainly from practical hydrocarbon exploration projects in Sichuan basin. The datasets is named as <b>***Carbonate-Sichuan-170G***</b> from the location and volume of the datasets. 22 types of carbonate frameworks are carfully divided.

After training by a <b>***ResNet101***</b> (parameters can be seen in <b>***Table 1***</b>, net framework can be seen in <b>***Fig. 2a***</b>), the <b>.pth</b> file to extract the corresponding heatmaps (CAM) on a lighter datasets, from <b>Qi Z., Hou M., Xu S., et al., A microscopic image dataset of Sinian carbonate from Dengying Formation on the northwestern margin of Upper Yangtze. Science Data Bank, 2020. (2020-07-31). DOI: 10.11922/sciencedb.j00001.00105., shorten as "MidDynuy"</b>. The pre-trained <b>***ResNet101***</b> took a role as a <b>Teacher Net</b> and distilled a light <b>Student Net</b> in <b>MidDynuy </b>(<b>***Fig. 3-4***</b>). The <b>Student Net</b> was a less than 800k ultra-lightweight model (original framework of MobileNetV3-Small can be seen in <b>***Fig. 2b***</b>; improved framework of MobileNetV3-Small can be seen in <b>***Fig. 3a***</b>; Details of improvements are displayed in <b>***Fig.3c-d***</b>). The <b>800k ultra-lightweight model</b> adapted the <b>***MobileNetV3-Small***</b> framework.

**<center>Table 1 Parameters of ResNets</center>**

<div align=center><img src=".\doc\img\resnet.jpg" width="800px" loc="center"/></div>

<div align=center><img src=".\doc\img\RandM.jpg" width="800px" loc="center"/></div>

**<center>Fig 2. Original Model Frameworks of (a) ResNet101 and (b) MobileNetV3-Small</center>**

<div align=center><img src=".\doc\img\imn.jpg" width="800px" loc="center"/></div>

**<center>Fig 3. Frameworks improved MobileNetV3-Small; (a) Total framework; (b) Illustration of one bottleneck; (c) Flow of bottlenecks; (d) Illustration of the skip; (e) Flow of contran necks</center>**

To investigating the potential deploying scenario, in addition to CAM, other image processing such as <b>glcm</b> enhancement was used in the final rendering.

**The workflow of this research is:**
<div align=center><img src=".\doc\img\workflow.jpg" width="800px" loc="center"/></div>

**<center>Fig 4. Workflow of this research</center>**

## The effect and comparison of generating heatmaps are shown below

**The following shows the original image to CAM heatmap to the GLCM enhanced CAM heatmap**

<div style="display: flex; align-items: flex-start;">
	<img src="./doc/img/img1.png" width="250rm"> 
	<img src="./doc/img/hot1.png" width="250rm"> 
	<img src="./doc/img/heat1.png" width="250rm"> 
</div> 

**Here are some more examples**

<div align=center><img src=".\doc\img\imgExample.png" width="1000px" loc="center"/></div>

<div align=center><img src=".\doc\img\heatExample.png" width="1000px" loc="center"/></div>



**Here's how to works in camera**

<div align=center><img src=".\doc\img\HeatMap_camera.gif" width="400px" loc="center"/></div>






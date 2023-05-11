# GeoCAM



<div align=center><img src=".\doc\img\heatmap.gif" width="300px" loc="center"/></div>

Our main tasks so far are to train a classification model, use this model to extract the corresponding heatmap dataset, and then use the modified less than 800k ultra-lightweight model for distillation, and apply the model to various application scenarios.In addition to CAM, other image processing such as glcm enhancement was used in the final rendering result

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






# Lamaah: A YOLO-Based Model for Multi-Scale Object Detection in UAV Systems
<img width="1608" height="513" alt="Cover" src="https://github.com/user-attachments/assets/5c055165-1118-43e0-a863-502fe04131e9" />

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Results](#results)
- [Evaluation](#evaluation)
- [Resources](#resources)

---

## overview
Object detection in UAV imagery is challenging due to scale variation and complex backgrounds, especially for small objects.

To address this, we propose **Lamaah**, an enhanced YOLOv26-based model that improves detection using three main components: **SVA**, **PBAP-Net**, and **ACMF-Head**. These modules enhance feature extraction, multi-scale fusion, and feature consistency.

* Experimental results on VisDrone2019 show that Lamaah achieves **47% mAP@0.5**, outperforming the baseline by **7.5%**, while maintaining efficient computation (**105.8 GFLOPs**, **15.2M parameters**).
  
## Objectives
Lamaah aims to improve multi-scale object detection in UAV images, focusing on small objects, through:
* Developing a YOLOv26-based model for a multi-scale object detection in UAV imagery.
* Integrating key modules (SVA, PBAP-Net, ACMF-Head) to enhance feature representation and multi-scale fusion.
* Evaluating on VisDrone2019 for high accuracy with efficient computation.

## Methodology
The proposed approach is based on the YOLOv26s baseline and enhanced with the following modules:
* **SVA:** Applies sparse attention to extract fine-grained features while reducing computational cost
* **PBAP-Net:** Performs bidirectional multi-scale fusion to preserve spatial details and improve small-object detection
* **ACMF-Head:** Uses adaptive weighting across scales to improve consistency and reduce prediction conflicts
<img width="1208" height="880" alt="image" src="https://github.com/user-attachments/assets/c8952d55-ae21-42b9-88d9-1f26d3316436" />

## Dataset 
In this study, we use the **VisDrone2019 dataset**, a widely used benchmark for UAV object detection. It consists of real-world images collected from 14 cities in China, covering diverse urban and rural environments.

 * The dataset contains **10,209 images** divided into training (6,471), validation (548), and testing (1,610) sets.

 * It includes 10 object classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor. Object sizes follow COCO-based categories (small, medium, and large). The dataset presents several challenges such as small objects, scale variation, occlusion, and dense scenes. All experiments are conducted using the official dataset split to ensure fair comparison.

Dataset link: [VisDrone2019 Dataset](https://github.com/VisDrone/VisDrone-Dataset), also provided  in data file
## Results
<img width="1305" height="1404" alt="Black and White Square Design Business Minimalist Logo (1)" src="https://github.com/user-attachments/assets/7aa944ef-06ac-4f6e-be15-44ac10c01fd4" />

## Evaluation 
In object detection, the key metrics for evaluating performance include Precision (P), Recall (R), Average Precision (AP), Mean Average Precision (mAP) , F1 Score and IoU. These metrics quantify how successfully a model recognizes and localizes objects in video or image frames. In addition, Floating Point Operations (FLOPs) and Giga Floating Point Operations (GFLOPs) are utilized to measure computational complexity. Also APsmall specifically measures the model’s performance on small-scale objects, which is particularly important in UAV scenarios where objects often appear at very small sizes.

## Resources 

<img width="2000" height="1106" alt="Black and White Square Design Business Minimalist Logo (2)" src="https://github.com/user-attachments/assets/fea2217b-93ba-4ace-9bba-7261add30ac3" />

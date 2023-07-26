# RoboChop: Autonomous Framework for Fruit and Vegetable Chopping Leveraging Foundational Models

### Introduction
This project presents a robotic system capable of autonomously chopping varied fruits and vegetables. The system combines computer vision techniques with robust robotic control to perform chopping tasks. This system contributes towards the development of autonomous cooking robots and identifies challenges that need to be addressed in future work.

### Methodology
In our system, we segment foods using a pre-trained segmentation model, Segment Anything Model (SAM), and identify and segment objects of interest in the scene with a YOLO v8 model. To address the challenge of identifying sliced fruits, we fine-tuned the YOLO model on a custom dataset of sliced and uncut fruits and vegetables.

Our approach consists of three key components:

1. Vision System
2. Action Primitive Library
3. Autonomous Loop

![image](https://github.com/Atharva4320/RoboChop/assets/55175448/a8020115-3612-41e4-84ef-7d5ae75e141a)

Detailed descriptions for each component can be found in our paper.

### Experimental Setup

We test and evaluate our proposed pipeline entirely in real-world experiments with a 7DOF Franka robot arm. The robot is equipped with a wrist-mounted Intel RealSense D415 camera, and a 3D printed blade tool. A detailed visualization of our experimental setup is shown below:

![image](https://github.com/Atharva4320/RoboChop/assets/55175448/536a6fdd-c477-4887-bafc-21fe3beca892)

### Results

We conducted a series of experiments to evaluate the robustness of our system, including testing the system on multi-class cluttered environments, assessing the reliability of the chop action primitive itself, and performing a set of multi-step, multi-class cluttered environment tasks.
![image](https://github.com/Atharva4320/RoboChop/assets/55175448/f5c5d57d-e5f3-4d08-8e94-e37433ff08e1)

### Conclusion and Future Work

Our study contributes towards the development of autonomous cooking robots and identifies challenges that need to be addressed. Future work should focus on enhancing the cut action success detector, incorporate tool changing to pick and place objects, and optimizing SAM’s inference time to handle larger numbers of objects.

#### Link to our [paper](https://arxiv.org/pdf/2307.13159.pdf)

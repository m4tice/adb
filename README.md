# Automated Driving Block week
Repository created for block week project: Autonomous Driving

![Alt text](https://github.com/m4tice/adb/blob/main/assets/banner04.png)

**SOFTWARES and FRAMEWORKS:**  
* [CARLA](https://github.com/carla-simulator/carla) > 0.9.10  
* [Conda](https://docs.conda.io/en/latest/)  
* Python

**IMPORTANT REQUIREMENTS:**  
* Python > 3.7.x  
* Tensorflow > 2.1.x  

**INSTRUCTIONS:**  
Install the Conda environment, which contains the necessary libraries by running the following commands:  

```
conda env create -f environment.yml
conda activate tf_gpu
```

After finishing CARLA installation, clone this repo and place it as follows:  

    .
    ├── ...
    ├── PythonAPI
    │   ├── blockweek_ad          
    │   ├── carla             
    │   ├── examples                      
    │   └── util                
    └── ...

# Stream A: Implementing end-to-end learning for autonomous driving and perform test drive on course 1
![Alt text](https://github.com/m4tice/adb/blob/main/assets/course1v2.gif)

**TASKS:**  
- [ ] Collect data  
- [ ] Process data  
- [ ] Build training network  
- [ ] Train models  
- [ ] Testing  

**GOAL:**  
<p align="center">
  <img src="https://github.com/m4tice/adb/blob/main/assets/E2E_result.gif">
</p>


# Stream B: Implementing Model Predictive Control for autonomous driving and perform test drive on course 2
![Alt text](https://github.com/m4tice/adb/blob/main/assets/course2v2.gif)

**TASKS:**  
Implement the Model Predictive Controller for vehicle in Carla based on the approach performed [here](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py).  

**GOAL:**  
<p align="center">
  <img src="https://github.com/m4tice/adb/blob/main/assets/MPC_result.gif">
</p>

# Credits
End-to-end learning: [naokishibuya](https://github.com/naokishibuya).  
Model Predictive Control: [AtsushiSakai](https://github.com/AtsushiSakai/PythonRobotics).

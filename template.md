# Applying CRISP-DM Methodology to the PuzzArm Project

**Document Purpose:**  
This document applies the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology to the PuzzArm project, structuring the AI/ML components (e.g., image identification for puzzle pieces, pose estimation, and imitation learning for arm control) across its six phases. CRISP-DM provides an iterative, non-linear framework for ML projects, emphasising business alignment and continuous refinement. This application serves as a checkpoint for the clustered AI course, mapping to ICTAII501 (designing AI solutions) and ICTAII502 (implementing ML models).

Use this as Assessment template, filling in project-specific details based on your work (e.g., Roboflow training/Arm training ).

**Project Recap:** PuzzArm is an AI-powered robotic system using Jetson Nano and xArm1S to solve a number puzzle (0-9 pieces), with dual-arm teleop (or similar method) for data collection. The ML focus is on vision-based detection, classification, and motion policies.

**Iteration Note:** CRISP-DM is cyclical—after Deployment, loop back to Business Understanding for refinements (e.g., adding new puzzles).

---

## Phase 1: Business Understanding
**Objective:** Define the problem, goals, and success criteria in business terms. Assess resources and risks.  

- **Business Problem:** Automate puzzle solving to create an educational robotics demo for expo demonstrations  and marketing events.
- **Data Mining Goals:** Develop models for piece detection (~70% accuracy), pose estimation (handling rotations), and arm control (50% pick-place success).  
- **Project Plan:** Timeline (4-6 weeks); resources (Jetson Nano, xArm1S, Roboflow). 
- **Risks:**
	- *Student input* I had limited time to complete the project due to external circumstances. 
	- *Student input* My dataset was limited and created under controlled conditions so there was a risk that the model wouldn’t generalise well. 


- ***Student Input:*** [Describe how you addressed the business need 100-200 words]  

For the final project, the main business goal was to create a small proof-of-concept demo to show how AI and robotics can work together. I originally wanted the system to recognise and pick up puzzle pieces but due to limited time, I focused on delivering a basic MVP rather than a fully working solution.

After discussing with my teacher and classmates, I confirmed that my goal was to build an MVP version of the PuzzArm system focusing only on imitation learning. The aim was to demonstrate that the model could learn from the joint positions I recorded during data collection and output movement predictions even if they were small.

The success criteria I set for myself were:
- Define and test the three robot arm poses (home, ready_to_grab, and ready_to_move).
- Collect the Grip dataset (images with puzzle piece and joint values).
- Collect the Navigation dataset (gripped piece with slight approach variations).
- Train the Grip model (and the Navigation model if time allowed).
- Run an initial test on the robot to see whether it could actually predict joint changes even if it only moves slightly.

*Mapping to Units:** ICTAII501 PC 1.1-1.2 (confirm work brief via CRISP-DM business phase).*

---

## Phase 2: Data Understanding
**Objective:** Collect initial data, explore it, and identify quality issues.  

- **Initial Data Collection:** 100-200 images of puzzle pieces/slots from top-down camera (via Jetson CSI), plus teleop videos (ROS2 bags) for joint states. Sources: Manual photos, Roboflow public datasets for augmentation.  
- **Data Description:** Structure (images: RGB, 224x224; labels: 0-9 classes; joints: 6D floats). Volume: ~5k samples post-augmentation.  
- **Data Exploration:** Use pandas/matplotlib for histograms (e.g., class balance: 10% per digit); identify issues (e.g., lighting bias via correlation plots).  
- **Student Input:** [From Part 1: Summarise your Roboflow dataset stats, e.g., "50 images/class; explored via Hello AI tools, found 20% rotation variance." Include a sample plot code/output.] 

I collected my dataset manually using a standard webcam placed on top of a monitor to capture a top-down view and took 31 images for the number 1 puzzle piece for the Grip dataset. Each image included the puzzle piece placed under the robot’s gripper in different rotations while the arm was set to the ready_to_grab state. At the same time I recorded the 6 joint values into a matching JSON file. I didn’t use any external tools like Roboflow or full data exploration libraries due to time constraints.

Instead of running statistical analysis or visualisation tools, I checked my images manually on-screen to make sure the puzzle piece was visible and positioned correctly for each capture. The images were taken with similar lighting at a similar distance from the camera and although I rotated the piece each time it was still mostly placed in the same area. At this stage I identified that this could limit the model’s ability to generalise to different positions or lighting.

I also started collecting a small Navigation dataset where the puzzle piece was held in the gripper and angled towards different slot positions but due to limited time and incomplete testing, I decided not to use this dataset in the final model training.

I didn’t generate any graphs or plots during this phase and I relied on visual inspection to confirm that the images and joint data were correctly paired before moving on to the next stage. If I had more time, I would’ve improved this phase by increasing variation in image location, lighting and arm position and by using simple exploration tools or plots to better understand dataset balance before training.

*Mapping to Units ICTAII502 PC 1.1-1.6 (analyse requirements and data attributes using CRISP-DM data phase).*  

---

## Phase 3: Data Preparation
**Objective:** Clean, transform, and construct the final dataset for modeling.  

- **Data Cleaning:** Remove duplicates/blurry images (OpenCV thresholding); handle missing labels via Roboflow auto-annotation.  
- **Feature Engineering:** Augment for rotations (0-360° via Albumentations); normalize images (0-1 scale); engineer joint deltas from teleop recordings.  
- **Final Dataset:** Train (70%): 3.5k samples; Val (20%): 1k; Test (10%): 500. Format: PyTorch DataLoader for Jetson training.  
- **Student Input:** [Detail your prep from thumbs classifier/Roboflow, e.g., "Applied flips and brightness augments to address orientation issues." Include before/after metrics, e.g., variance reduction.]  

For this stage most of the dataset preparation was handled automatically through my camera.py script. When I took each image the script saved it and also recorded the matching joint values into a JSON file using the same filename. This was done so that each image was correctly paired with its joint data without having to manually organise anything.

I didn't use any automated data cleaning or augmentation tools such as Roboflow. Instead I visually inspected the images during collection and deleted any that looked blurry or didn't properly show the puzzle piece. I also didn't attempt extra brightness or positional augmentation beyond physically rotating the piece during collection.

During training I used basic PyTorch transforms inside my Colab notebook (such as resizing the images to 224×224 pixels, converting them to tensors and applying normalisation). I didn’t manually create separate folders for training and validation but the dataset split was handled automatically in the Google Colab notebook using PyTorch’s DataLoader. I didn’t create a separate test set due to the small dataset size.

The joint deltas weren't manually engineered and were calculated during training as part of the model’s output layer. If I had to do this again I would;ve improved this phase by adding proper dataset splitting (including a separate test set), introducing more variation using data augmentation (such as different lighting, positions and more rotation angles) and using automated tools to better assess dataset quality before training.

- **Mapping to Units:** ICTAII502 PC 2.1-2.4 (set parameters, engineer features per CRISP-DM prep phase).  

---

## Phase 4: Modeling
**Objective:** Select and apply ML techniques, tuning parameters.  

- **Model Selection:** - *Student input* - Name, use etc

I used a supervised imitation learning approach and created a regression model called GripNet. The model was built using CNN architecture, specifically ResNet18 with pretrained ImageNet weights as the backbone. I replaced the original classification layer with an identity layer and added a fully connected regression head so the model could output six joint delta values instead of class labels. I also planned to create a second model called NavNet using the same approach but due to limited time I only completed training the GripNet model.

- **Techniques Applied:** - *Student input* - eg, Supervised training 

I used supervised learning where each input image was paired with the robot’s current joint values and labelled with the expected movement adjustment (delta values). I trained the GripNet model for 20 epochs using the Adam optimiser and Mean Squared Error (MSE) loss. I didn't perform hyperparameter tuning due to time constraints.

Instead of using a separate validation or test dataset I carried out model evaluation through live robot testing using a Python script. The goal was to confirm whether the model could make small adjustment predictions rather than achieving precise pick-and-place performance.

- **Model Building:**  - *Student input* - eg, Train detection first (output: boxes/classes); then policy (input: cropped image + joints; output: 6D deltas). Export to TensorRT.  

During training each image was resized to 224×224, converted to a tensor and normalised using PyTorch transforms. In the forward() method, I used ResNet18 to extract image features then combined these with the current joint values and passed them through a fully connected layer to output the six predicted joint deltas.

I followed pseudocode provided by my teacher and used ChatGPT to help understand and complete the model structure as some parts were pretty complex. This allowed me to correctly implement the ResNet18 backbone and the forward pass logic.

I only trained the GripNet model in Google Colab due to time constraints and didn't complete the Navigation model. After training I tested GripNet using my Python script which confirmed that it could produce small joint adjustment predictions in a live trial.

*Mapping to Units ICTAII502 PC 3.1-3.5 (arrange validation, refine parameters via CRISP-DM modeling).*  

---

## Phase 5: Evaluation
**Objective:** Assess model performance against business goals; review process.  

- **Model Assessment:** - *Student input* - eg, Metrics: Detection,  Policy success rate, pick-place trials,  Use confusion matrix for classes.

To evaluate the model I ran a live test at home using my Python inference script and manually positioned the puzzle piece under the gripper in the same setup used during data collection. When I executed the test the robot did move slightly in response to the model’s predictions. This confirmed that the model learned some relationship between the image and the joint deltas.

During training in Google Colab the model was trained for 20 epochs using the MSE loss function. The training loss improved from 177.29 in epoch 1 to 10.84 in the final epoch while the validation loss decreased from 139.27 to 12.88 by the end of training. The validation curve was not fully stable and occasionally had large spikes in epochs 7–10 which was expected due to the very small dataset and lack of variation. Despite instability the model still demonstrated that it could learn a relationship between the images, current joint positions and the predicted deltas. I repeated the test in class using the same script and setup and observed identical results. This confirmed that the model was behaving consistently but also showed that improvement would require a larger and more varied dataset or model tuning.

- **Business Criteria Check:** Does it enable full puzzle solve <5 min?  - *Student input* 

My original goal was to build an MVP to demonstrate how AI could be applied to robotic control using imitation learning. Although the model didn't perform a full puzzle-solving action, or even a successful grip action, it successfully demonstrated the key AI concept of predicting movement based on learned examples. The movement was small but still showed that the system was processing the image and generating a joint change which partially meets the demonstration objective.

Due to dataset limitations and time constraints, the model didn't achieve full piece detection, navigation or a complete pick-place action. It also did not solve the puzzle within the ideal target time but it still met the basic MVP goal of proving that imitation learning could drive robotic movement.

- **Process Review:** Data quality issues? (e.g., rotations fixed via augments). Next iteration: - *Student input* 

During this phase of the project I realised that the main limitation was data quality and diversity. Most images were captured under the same lighting and position which restricted how well the model could generalise. Although I rotated the puzzle piece during collection, I now understand that I should have introduced more variation such as different locations, distances from the camera, lighting conditions and more examples of the piece positioned away from the gripper.

For the next iteration, I would:
- Expand the dataset with more variation in position, orientation, and environmental conditions
- Recollect both Grip and Navigation datasets more comprehensively.
- Create a proper validation and test split rather than relying on live testing.
- Train the Navigation model and apply consistent evaluation methods.
- Attempt hyperparameter tuning or train for longer.

Despite the limitations this phase confirmed that the model was able to learn basic movement prediction and acted as a successful technical proof of concept.

*Mapping to Units ICTAII502 PC 5.1-5.6 (finalize evaluations, document metrics per CRISP-DM eval phase); ICTAII501 PC 3 (document design outcomes).*  

---

## Phase 6: Deployment
**Objective:** Plan rollout, monitoring, and maintenance.  

- **Deployment Plan:** *Student input* - eg, Deploy on Jetson. 

I didn't fully deploy the model onto the Jetson Nano but innstead I ran it using a local Python script (test_grip_model.py) on my computer connected to the robotic arm. The script loaded the trained grip_model.pth, set the arm to ready_to_grab, captured a webcam image, processed it using the same model transforms, retrieved the current joint values (get_joints) and then applied the model’s predicted deltas using apply_deltas(scale=0.2, duration_ms=1500).

The script ran three attempts and I manually repositioned the puzzle piece between each one. This acted as a simple demonstration method rather than a full deployment. If I continue this project I would move the model to the Jetson Nano for standalone execution and explore model optimisation.

- **Monitoring:** *Student input* - eg, Log pickup  time (<100ms); retrain quarterly with new demos.  

Monitoring was done through manual observation and console output. The script printed the current joint angles and predicted deltas and I watched the arm to check if any movement occurred. The movements were consistently small, matching the model’s limited accuracy and small dataset. No automated tracking or metric logging was used. In future I would add basic logging such as movement success rates, prediction timing and performance tracking.

- **Business Reporting:** *Student input*  - Demo video; report ROI (e.g. What it can do for the time invested). Maintenance: Version models in GitHub/Gitlab]  

For this submission I used a demonstration of the script-driven test as evidence, showing the robot responding to the model’s predictions. A video can be viewed [here](https://www.youtube.com/watch?v=qkX_V8NIIWA). I also included the GripNet model code and the test_grip_model.py script in my documentation to show how the system works end-to-end. 

In future versions, I would track trained models and updates using GitHub to help compare improvements over different iterations. Considering the limited timeframe and resources, I believe the project delivered good value as an MVP learning outcome as it successfully demonstrated the core AI concept even without full functionality.

*Mapping to Units ICTAII501 PC 2 (design for deployment); ICTAII502 PC 4.1-4.5 (finalize test procedures).*  

---

## Overall Reflection and Iteration Plan
 **Next Steps:** *Student input* - What do you need to do next to achieve the project.  200 -400 words + code samples if required.

This project was my first time combining my own ML model with a real robotic arm and it taught me a lot about how many pieces have to work together such as data collection, model design, training and live testing. I felt proud that I was able to:
- Collect my own grip dataset using a webcam and joint recordings.
- Build and train a CNN-based regression model (GripNet) using ResNet18.
- Write a test script that connected the camera, model, and arm so I could see real movement based on AI predictions.

Even though the movement was small, it still showed that the model had learned a relationship between the images, the joint values and the target deltas. Running the test both at home and again in class gave consistent results which confirmed that the behaviour was repeatable and not random.

The main limitations I noticed were:
- The dataset was small and captured under very controlled conditions.
- I only fully trained the Grip model and not the Navigation model.
- I relied on live testing rather than a proper held-out test set for evaluation.

If I continue this project, my next steps would be to:
- Recollect a much larger and more varied dataset (different positions, distances, lighting and starting joint states).
- Introduce proper data augmentation (brightness shifts, slight translations, rotations and camera variations) to help the model generalise better.
- Fully train the Navigation model and evaluate both models with proper train/validation/test splits.
- Implement automated monitoring and performance metrics. 

Even though the final behaviour was limited, I achieved my MVP goal of demonstrating that an imitation learning model I trained myself could drive real robot motion. I now have a much clearer understanding of what would be required to improve this system and am motivated to explore further development as part of self-directed learning at home at my own pace. 
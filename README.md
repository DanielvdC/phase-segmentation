# Phase-segmentation
Project to define surgical phases for the laparoscopic cholecystectomy procedure.

This project includes a pre-trained model, which is the EfficientNetB3 and trained on about 24.000 images. The trained models can be found in ./Output/models/ and includes three variations, based on the amount of frames consideren during training.
To retrain the model with your own dataset, use the ./scripts/efficientnetb3.py file.

After predictions are made, you can choose a general or specific optimisation method. The general method (.scripts/post_processing_general.py) uses set rules to enhance performance. These methods are defined by a trail-and-error approach and do not need any further alterations.
The specific method (.scripts/post_processing_specific.py) includes a wide range of possible rules, but still need to be applied. This method generally produces better results than the general method, but is more time-intensive.

With these scripts, you can generate predictions for the surgical phase timestamps for the laparoscopic cholesystectomy procedure. In order to do so:
1. Clone the repository into your environment
2. Install necessary packages with:
<br> pip install -r requirements.txt
3. Either retrain with your own images with the ./scripts/efficientnetb3.py script
4. Or add your videos to the ./Data/videos/testset folder
5. Run the ./scripts/video_predictions.py script
6. Choose your post-processing method (either general or specific)

Output will be a plot.

# Author
Daniël van den Corput
<br> danielvdcorput@gmail.com

# Purpose
This project was created for Incision, see https://www.incision.care/

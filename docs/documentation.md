# DeepCream - code documentation

In this documentation file we will explain the approach and development process of
each module in short. The technical details of the execution can be found in the code
which has lots of comments closely explaining the individual lines of code.

## Cloud detection
The original idea for the cloud detection was using the OpenCV library together
with a smart algorithm to filter the clouds out by their HSV values. However, this method
soon turned out to be too unreliable on its own with is why we trained an AI to assist it.

The AI is based on the U-Net Model which proved to be a good choice, delivering promising
results from the start. The problem with this approach was that it required
training data where the cloud masks would have to be made by hand which would have
taken multiple hours.

Luckily we were able to automate this process using [Apeer](https://www.apeer.com/home/) which
is an automated image analysis originally meant for biotechnology but which
is also a great solution for segmenting clouds in satellite images. With this tool we only
had to annotate about a dozen photos and were able to generate the remaining masks easily and quickly.

Through this unexpected success the AI became so good that it rendered the HSV
analysis basically useless, which allowed us to cut it from the cloud detection module,
reducing the processing time per image by over 70%!

![](cloud detection/unet-model.png)
Source: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## Cloud analysis

## Classification

## Pareidolia

## Database

## main.py
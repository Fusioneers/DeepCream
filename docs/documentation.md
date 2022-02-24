# DeepCream - code documentation

In this documentation file we will explain the approach and development process
of each module in short. The technical details of the execution can be found in
the code which has lots of comments closely explaining the individual lines of
code.

## Cloud detection

The original idea for the cloud detection was using the OpenCV library together
with a smart algorithm to filter the clouds out by their HSV values. However,
this method soon turned out to be too unreliable on its own with is why we
trained an AI to assist it.

The AI is based on the U-Net Model which proved to be a good choice, delivering
promising results from the start. The problem with this approach was that it
required training data where the cloud masks would have to be made by hand
which would have taken multiple hours.

Luckily we were able to automate this process
using [Apeer](https://www.apeer.com/home/) which is an automated image analysis
originally meant for biotechnology but which is also a great solution for
segmenting clouds in satellite images. With this tool we only had to annotate
about a dozen photos and were able to generate the remaining masks easily and
quickly.

Through this unexpected success the AI became so good that it rendered the HSV
analysis basically useless, which allowed us to cut it from the cloud detection
module, reducing the processing time per image by over 70%!

![](cloud_detection/unet-model.png)
Source: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## Scientific application

The scientific application is split into two distinct parts: The analysis,
which separates the clouds on the provided image and computes a variety of
properties for each one. The classification on the other hand part takes these
features and computes how likely the different cloud types are for each of
them.

### Cloud analysis

The first part of the analysis consists of the separation of the clouds. These
are mostly considered independent. The second part of the analysis identifies a
couple of features that can indicate the type of the cloud. These features
consist of shape based features, for example the elongation or the convexity,
and texture based features, like transparency and sharp edges.

### Classification

At first, we thought about implementing a clustering algorithm to identify the
different clouds. However, this would have had the disadvantages of consuming a
lot of computation time and resulting in barely predictable results. So instead
we computed the classification by comparing the properties gathered by the
analysis against a couple of cloud types previously defined. This allows us to
quickly customise and specify new cloud types and correct classification
mistakes made by the program, while maintaining a high performance.

## Pareidolia

In this part of our project, we attempted to simulate the phenomenon of
pareidolia. Pareidolia is the tendency for perception to impose a meaningful
interpretation on a nebulous stimulus, usually visual, so that one sees an
object, pattern, or meaning where there is none
(Source: [Wikipedia](https://en.wikipedia.org/wiki/Pareidolia)).

To achieve this, we used the MPEG-7 dataset to train a neural network that can
classify shapes of more than 50 different everyday objects using binary masks.
To train the AI we
used [Teachable Machine](https://teachablemachine.withgoogle.com/).

![](pareidolia/MPEG-7_01_overview.png)
Source: https://www.researchgate.net/profile/Joao-Nunes-29/publication/245251999/figure/fig4/AS:668849762492443@1536477618671/The-MPEG-7-Core-Experiment-CE-Shape-1-dataset.png

## Database

Because we wanted to be able to restart the program multiple times, a proper
database was needed. At first, we wanted to implement a relational database
because of the ease of sqlite3 in python, but then we quickly realized that a
document based database would yield far better results. As systems like MongoDB
are not installed on the ASTRO PI, we implemented a database ourselves. More
concretely is it a wrapper around a folder structure described more detailed in
the docstrings of the module. It handles all in- and output to the file system,
allowing a concise interface in the other modules.

Because the allowed space of 3GB is not enough to store all images in full
resolution, we implemented a smart algorithm to select images which are less
relevant to our program and compress and delete them if necessary. The
so-called quality determines how confident the algorithms are about an image
(i.e. if they are very sure that a cloud is of a specific type or looks very
similar to a specific object). Note that only the original images taken by the
camera are considered for compression or deletion. No output of the scientific
or artistic approach is deleted, and no image is compressed or deleted which is
not fully analysed. Below you can see a flow chart of this algorithm:

![](storage_management/storage_management_algorithm.png)

Everytime an image is to be saved, the remaining space is checked. If there is
not enough space to store the image, then the free_space method takes place. If
there are any not deleted images, then the ratio of the already compressed
images to the neither compressed nor deleted ones is determined. This ensures
that the images, for which the classification and pareidolia are very
confident, are not compressed unless there is no other choice. If 85% are
compressed, then the image with the lowest quality is deleted, otherwise
compressed. If there are no not deleted images with a quality, but images
without, then a signal i.e. an error is raised which urges the program to slow
the taking of images down until more qualities are computed. In the unlikely
scenario, that the csv files and masks use more space than 3GB, the program is
stopped.

## main.py and deepcream.py

The main.py and the deepcream.py control the other modules and ensure a smooth
execution. To be able to do so they monitor and limit the execution time as
well as the CPU temperature and number of active threads. They make sure that
both stay under a certain threshold. If the CPU temperature for example gets to
hot, the program will make a 60s pause to cool it off. If the number of active
threads ever got too big the program would stop them, restart and continue
where it left off. This is only possible thanks to the elegant interplay
between these modules and the database.

There are two small "issues" wich aren't really problems but which should be
noted:

1. When the deepcream module is restarted, the camera connection is closed and
   opened again. In case the get_orig thread hasn't finished yet, it raises a
   single exception, that the camera connection is closed. This error is caught
   and has no effect.
2. In some very rare cases there might be an exception after the program which
   say something like "terminate called without an active exception". This, as
   extensive testing has shown, doesn't have any effects at all and can be
   ignored. We know that the threading is the origin of the "error" but we
   aren't really sure what's causing it.

So to summarize the program will handle all exceptions, log them and if
necessary restart the program. We've tested this in great detail and are
confident the program will be able to fulfill its purpose with ease.

If you do have questions or remarks please don't hesitate to contact us, we
will be happy to help: <info@fusioneers.space>

# DeepCream

**Note:** Since this project is intended to be submitted to the Astro Pi competition in the form of a .zip file,
we would like to point out that a [GitHub repository](https://github.com/Fusioneers/DeepCream) is available.
We also invite you to visit [our website](https://www.deepcream.eu/), where you can find more information
about our team and the work on the project.

If you have any problems, please contact us at <info@fusioneers.space>.

### Announcements

#### Shooting for the stars

We are excited to announce that the first release of DeepCream is almost complete.
We will be ready to launch by **23:59 CET** / **22:59 GMT** on **24th February 2022**.

#### First successful test

A full 3-hour test suggests that DeepCream runs stably and produces meaningful results.
We will conduct further tests to ensure that the application is prepared for all eventualities. 

## Welcome to DeepCream, an Astro Pi 2022 project!

<!--
What makes your project stand out?
connecting multiple AIs
-->

### When gazing into the sky...
...there are beautiful objects that catch one's eye and may keep it for a while: The clouds.

In this project we deal with the clouds.
We find them all, we analyze them down to the smallest detail,
and we interpret artistic forms into them.

The interplay of three artificial intelligences, two of them being artificial neural networks,
under control of the [`main.py`](main.py) enables DeepCream to perform these tasks efficiently.

DeepCream is registered at the 2022 [Astro Pi competition](https://astro-pi.org/) and has so far passed the first round.
Astro Pi is an ESA Education project run in collaboration with the Raspberry Pi Foundation.

### Getting started

#### Installation

When having [git](https://git-scm.com/) installed, you can easily clone DeepCream running the following command in the
directory where you want the application to be installed.

```bash
git clone https://github.com/Fusioneers/DeepCream.git
```

Otherwise, you can download the repository as a .zip file and extract it.

#### Usage

You can run DeepCream from the command line using the following command:
```bash
python3 main.py
```

#### Customization
You can customize your copy of DeepCream by modifying the [DeepCream/constants.py](DeepCream/constants.py).
For instance, you can turn TPU support on and off, customize file formats,
change the location of your database or edit cooldown delays.

Note that the GitHub version
of [DeepCream/constants.py](DeepCream/constants.py) is configured for the Astro Pi competition.

#### TPU support
DeepCream supports for Edge TPUs from Coral.
To get started using a TPU, just connect it to the computer and run DeepCream.
Make sure the variable `tpu_support` in [DeepCream/constants.py](DeepCream/constants.py) is set `True`.

More information can be found under [docs/documentation.md](docs/documentation.md).

## Features

### Core Features

* Cloud detection
* Cloud analysis
* Cloud classification
* Artistic interpretation of cloud shapes

### Other Features

* Database management
* Detailed logging
* TPU support

## Requirements

This project is build to operate on a Raspberry Pi running ESA's Flight OS with **Python 3.7.3**.

When running on a Raspberry Pi or similar device we recommend [using a TPU](#tpu-support).

### Python Interpreter

* Version 3.7.3

### Operating Systems

* Linux
* macOS
* Windows

### Required Libraries

* Skyfield
* picamera
* colorzero
* gpiozero
* GDAL
* numpy
* SciPy
* TensorFlow, TensorFlow Lite, and PyCoral
* pandas
* logzero
* Keras
* matplotlib
* pisense
* Pillow
* opencv
* exif
* scikit-learn
* scikit-image
* reverse-geocoder

## Implementation

This is a brief description of how each part of the project interacts.
A detailed description of the functionality of the project can be found in
the [Documentation](docs/documentation.md).
Also, there are docstrings in the code that explain the precise technical implementation.

DeepCream uses two deep learning models. The first determines the position of the clouds and puts a corresponding mask
on the images.

We made intense use of OpenCV and Pillow to extract detailed features from the images.

Also, we use an algorithm to determine cloud types by comparing computed features of the clouds (e.g. shape, thickness)
to criteria defined in a [cloud types file](DeepCream/classification/cloud_types.py).

The second model it was created using TeachableMachine and trained with parts of
the [MPEG-7 dataset](https://en.wikipedia.org/wiki/MPEG-7).


<!--
Optional content:
## Known errors
## FAQ
## Copyright and licensing information
-->


Happy cloud observation!


---
**Fusioneers** ([@Fusioneers](https://github.com/Fusioneers)) 2022

* **Kevin Kretz** ([@theKevinKretz](https://github.com/theKevinKretz))
* **Daniel Meiborg** ([@DanielMeiborg](https://github.com/DanielMeiborg))
* **Paul Maier** ([@C0mput3r5c13nt15t](https://github.com/C0mput3r5c13nt15t))
* **Lukas Pottgiesser** ([@Lukas-Pottgiesser](https://github.com/Lukas-Pottgiesser))

* Xinyue (Lucia) Guo ([@aiculguo](https://github.com/aiculguo))
* Jannis Sauer

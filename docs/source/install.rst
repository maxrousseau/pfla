Installation and Usage
======================

Requirements and Dependencies
-----------------------------

-   Python 3.5 (or higher)
-   Python packages: numpy, pandas, pillow,

Installation
------------

To install with pip:

.. code-block:: shell

    pip install -r requirements-pytorch.txt\ # pytorch for CPU
         -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt # other dependencies
    pip install pfla

Usage
-----

.. code-block:: shell

    usage: pfla [-h] [-d] [-l] [-m] [-o OUTPUT] [-v] path

    PFLA: python facial landmark analysis. This program will read the image(s)
    given as input and can apply a face detection algorithm, landmark placement
    and computation of metrics. The results are returned as a text stream.

    positional arguments:
      path                  path to the image or directory of images

    optional arguments:
      -h, --help            show this help message and exit
      -d, --detect          detect faces and output bounding box
      -l, --landmark        annotate detected faces and output coordinates
      -m, --metrics         compute metrics and output results
      -o OUTPUT, --output OUTPUT
                            specify output filename and format/filetype of the
                            data
      -v, --verbose         increase output verbosity

    AUTHOR: Maxime Rousseau LICENSE: MIT

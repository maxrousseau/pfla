Overview
========


Image Processing
----------------

This program takes as inputs facial image(s) (supported formats: jpg, png
tiff, bmp) for initial processing and prepare for landmarking and analyis. The
image(s) are then processed as follows: facial detection with MTCNN, 68
landmark face annotation, computation of metrics.

Output
------

By default full image processing is done and outputed into to file (default:
``out.csv``). The user may specify what output is desired as well as the desired
output file (supported formats: csv, pkl, h5, xlsx). See
usage for details.


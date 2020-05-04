#from img_prep import ImgPrep
#import numpy as np
#newim = ImgPrep('../tests/data/', GRAY=True, EXT='jpg')
#newim.prepare_dir()

from face_detect import FaceDetect
import numpy
from PIL import Image
im = numpy.array(Image.open('../tests/data/testpic.jpg'))
nface = FaceDetect(im)
box = nface.mtcnn_box()

from annotate import FaceAnnotate
ante = FaceAnnotate(im, box)
myldmk = ante.get_ldmk()

from metrics import Metrics
met = Metrics(myldmk)
met.compute_metrics()


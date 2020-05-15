# Change Log

Overview of changes and planned features.

## TODO

Devops:

- setuptools and PyPI Ok (no autoinstall of packages until facenet-pytorch and
  face-alignment are replaced)
- improve documentation (not detailed enough bare)
- code profiling (pfla is very slow)

Source:

- logging/progress: better logging (encapsulate within fcn/classes
	instead of in cli.py and progress with tqdm
- more elaborate testing (see test-cov pytests)
- (model implementations)
	* mtcnn.py
	* landmarkmodel.py (Face Annotation Network/FAN)
	* GPA/mean shape implementation
- cli.py: -g grayscale (no use for it currently), remove?
- ability to save the output img to a desired directory (pillow)
- refactor with flake8 and black

Features:

- GPA
- Draw and save input images
- Asymmetry metrics
- Profile analysis

## v1.0.1 further code sanitization

The only features that will be added for this release are the alignment with
GPA and the ability to save the annotated images.

## v1.0.0 Rewriting PFLA

The new version of pfla will no longer automatically conduct statistical
analyses on the groups. The reason for this is that I aim to make the program
more simple, flexible and modular. These statistical analyses can easily be
performed after the images have been processed. This program will be solely
focused on the processing of facial image to produce data which can be used for
scientific purposes.

NOTE: resize input and make sure that colors are RGB formats supported: jpg,
png, bmp, tiff. Export file format support hdf5, csv, xslx, pickle. GPA not
supported for now.

### Goals

1. Making pfla pure python

2. Updating algorithms
	Add MTCNN facial detection and pytorch facial analysis

3. Simplifying use (cli)

```bash
$ pfla [options] (path)
	-d detect
	-l landmark
	-m metrics
	-o outfile and format
```

4. Limit dependencies

- numpy
- pillow
- pytorch
- currently (face\_alignment and mtcnn), to be implemented

## v.0.1.1 and previous

See commits and JOSS review



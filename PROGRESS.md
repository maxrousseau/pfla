# Progress

Overview of changes and planned features.

## v0.1.2 Rewriting PFLA

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

## TODO

- logging/progress: better logging (encapsulate within fcn/classes
	instead of in cli.py and progress with tqdm
- tests for each module
- setup finalization
- documentation (sphinx)
- travis CI
- (model implementations)
	* mtcnn.py
	* landmarkmodel.py (Face Annotation Network/FAN)
	* GPA/mean shape implementation
- cli.py: -g grayscale

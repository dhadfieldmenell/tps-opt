tps-opt
=======

High-Throughput Library for Fitting Thin Plate Splines

Dependencies
============
python2.7, scipy0.14, numpy1.8.1, gfortran, cuda6.0, PyCuda2013.1.1, scikits.cuda0.5.0, cmake, boost-python

Install Instructions
====================
You can install PyCuda, numpy and scipy with pip install to get the latest versions.

Install Cuda6.0
http://www.r-tutor.com/gpu-computing/cuda-installation/cuda6.0-ubuntu
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html

Install latest scikits.cuda from source (version available through pip doesn't have integration for the batched cublas calls yet).

$ git clone https://github.com/lebedov/scikits.cuda.git
$ cd scikits.cuda
$ python setup.py install 

The last line may need to run as root.

Check out the tps-opt repo and build the additional cuda functionality
$ git clone https://github.com/dhadfieldmenell/tps-opt.git
$ cd tps-opt/tpsopt
$ cmake .
$ make

It has been tested with the RLL overhand-knot tying demonstration dataset. Obtain a copy from https://www.dropbox.com/s/wnt3j42jp5solr8/actions.h5. 

To check the build, cd to tps-opt/tpsopt and run the following

dhm@primus:~$ cd src/tps-opt/tpsopt/
dhm@primus:~/src/tps-opt/tpsopt$ python precompute.py ../data/actions.h5 --replace --verbose
precomputed tps solver for segment failuretwo_5-seg02
dhm@primus:~/src/tps-opt/tpsopt$ python batchtps.py --input_file ../data/actions.h5 --test_full
running basic unit tests
UNIT TESTS PASSED
unit tests passed, doing full check on batch tps rpm
testing source cloud 147
tests succeeded!
dhm@primus:~/src/tps-opt/tpsopt$ python batchtps.py --input_file ../data/actions.h5
batchtps initialized
Running Timing test 99/100
Timing Tests Complete
Batch Size:                     148
Mean Compute Time per Batch:    0.0724914503098
BiDirectional TPS fits/second:  2041.62007199

You should see results analogous to those above. That example was run with an NVIDIA GTX770.
You can set the default behavior for the batchtps main in batchtps.parse_arguments.
The default parameters for the TPS fits is found in defaults.py. 

Notes
=====
In the scripts directory is Robert Kern's kernprof.py. It is a line-by-line profilier for python. The required packages can be installed from https://pythonhosted.org/line_profiler/. Running batchtps through that with --sync will let you see a (slower due to profiler overhead, and synchronous execution) breakdown of the timings for the gpu kernel calls. To run this, you will need to comment in the @profile decorators on the functions you wish to time. Example output for timing batch_tps_rpm and the various functions calls is included in EXAMPLE_TIMING.md. If the library is running slowly, check your results against that as a first step to gauge where the problem is.
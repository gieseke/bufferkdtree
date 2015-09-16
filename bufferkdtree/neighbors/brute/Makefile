
all: clean all_wrappers

all_wrappers:
	python setup.py build; cp build/lib.*/neighbors/brute/*.so .; cp build/src.*/swig/*.py .; cd swig; rm -f *wrap.c; cd ..; rm -rf build;

clean:
	rm -rf build; rm -f *.o; rm -f *.pyc; rm -f *~; rm -f swig/*~; rm -f *.so; rm -f wrapper_cpu_double.py; rm -f wrapper_cpu_float.py; rm -f wrapper_gpu_opencl_double.py; rm -f wrapper_gpu_opencl_float.py
	rm -f *.so

.. -*- rst -*-

Changes
=======

Release 1.4 (November 2019)
-----------------------------
* Updated documentation 
* Python3 support

Release 1.3 (November 2016)
-----------------------------
* Updated documentation (which now also includes a documentation of the C/OpenCL source code)
* Adapted/Added examples
* Adapted installation process (e.g., CPU based k-d trees can now be built without OpenCL support)
* Small code modifications and bugfixes

Release 1.2 (July 2016)
-----------------------------
* Added support for double precision
* Added Python 3 support
* Small bug fixes
* Fixed memory leak (`GH1 <https://github.com/gieseke/bufferkdtree/issues/1>`_)

Release 1.1.1 (December 2015)
-----------------------------
* Updated documentation

Release 1.1 (December 2015)
-----------------------------
* Fixed wrong parameter assignment in 'kneighbors' method of both neighbors/kd_tree/base.py and neighbors/buffer_kdtree/base.py
* Added Multi-GPU support to brute-force approach (for benchmark purposes)
* Adapted parameter settings for buffer k-d tree implementation
* Added benchmark example

Release 1.0.2 (September 2015)
------------------------------
* Adapted building process

Release 1.0.1 (September 2015)
------------------------------
* Adapted building process
* Fixed small bugs

Release 1.0 (September 2015)
----------------------------
* First major release
* Python wrappers for three implementations ('brute', 'kd_tree', 'buffer_kd_tree')
* Performance improvements for both kd-tree based implementations
* Multi-OpenCL-Device support for 'buffer_kd_tree' implementation
* Large-scale construction for 'buffer_kd_tree' implementation
* Multi-OpenCL-Device support for query phase (queries are processed in chunks)
* Added Sphinx documentation

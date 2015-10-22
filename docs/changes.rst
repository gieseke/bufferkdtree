.. -*- rst -*-

Changes
=======

Release 1.1.0 (November 2015)
----------------------------
* Fixed wrong parameter assignment in 'kneighbors' method of both neighbors/kd_tree/base.py and neighbors/buffer_kdtree/base.py
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

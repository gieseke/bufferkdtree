.. -*- rst -*-

Changes
=======

Release 1.0 - (under development)
-----------------------------------
* First major release
* Python wrappers for three implementations ('brute', 'kd_tree', 'buffer_kd_tree')
* Several small performance improvements for both the 'kd_tree' and the 'buffer_kd_tree' implementation
* Large-scale construction possible for 'buffer_kd_tree', i.e., in case the training patterns on the OpenCL device, then the patterns are processed in chunks (interleaved copy/compute)
* Multi-OpenCL-Device support for accelerating the query phase (query patterns are processed in chunks)


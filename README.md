# nms-simd

One of the biggest latency sources in object detection is non maximum suppresion. nms-simd uses SIMD instructions to perform non maximum suppresion minimum 2x to 9x faster than other implementations (tensorflow, opencv, torchvision etc). 

Please note that currently this is an experimental library. Only AVX2 instructions are supported, NEON and SSE4 implementations are on roadmap.

## Bencmarks 

We are benchmarking results from two different models ([yolov5](https://github.com/ultralytics/yolov5) and [FasterRCNN](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn)), six different images ([dog.jpg](https://github.com/pjreddie/darknet/blob/master/data/dog.jpg    ), [eagle.jpg  ](https://github.com/pjreddie/darknet/blob/master/data/eagle.jpg  ), [giraffe.jpg](https://github.com/pjreddie/darknet/blob/master/data/giraffe.jpg), [horses.jpg ](https://github.com/pjreddie/darknet/blob/master/data/horses.jpg ), [kite.jpg](https://github.com/pjreddie/darknet/blob/master/data/kite.jpg   ), [person.jpg ](https://github.com/pjreddie/darknet/blob/master/data/person.jpg )) and six different algorithms (nms-simd, tensorflow, torchvision, [faster-nms](https://www.pyimagesearch.com/faster-non-maximum-suppression-python/), opencv, mmcv). Current results are taken from a Ryzen 2600 machine.

### FasterRCNN Results

```
------------------------------------------------------------------------------------------------
Benchmark                                                Time             CPU   Iterations
------------------------------------------------------------------------------------------------
nms_simd/dog.jpg                                        1105 ns         1105 ns       617743
nms_simd/eagle.jpg                                      1020 ns         1020 ns       650623
nms_simd/giraffe.jpg                                    1036 ns         1036 ns       671031
nms_simd/horses.jpg                                     1175 ns         1175 ns       577970
nms_simd/kite.jpg                                       1512 ns         1512 ns       456069
nms_simd/person.jpg                                     1105 ns         1105 ns       626307
tensorflow_nms/dog.jpg                                 61193 ns        61193 ns        11106
tensorflow_nms/eagle.jpg                               60303 ns        60304 ns        11297
tensorflow_nms/giraffe.jpg                             61006 ns        61006 ns        11364
tensorflow_nms/horses.jpg                              61470 ns        61469 ns        11156
tensorflow_nms/kite.jpg                                62364 ns        62363 ns        10997
tensorflow_nms/person.jpg                              61560 ns        61559 ns        11249
vision_nms_with_score_threshold/dog.jpg                55505 ns        55870 ns        12092
vision_nms_with_score_threshold/eagle.jpg              46633 ns        46972 ns        14961
vision_nms_with_score_threshold/giraffe.jpg            55158 ns        55528 ns        12499
vision_nms_with_score_threshold/horses.jpg             56591 ns        56953 ns        11059
vision_nms_with_score_threshold/kite.jpg               57195 ns        57563 ns        11775
vision_nms_with_score_threshold/person.jpg             55802 ns        56158 ns        12268
vision_nms/dog.jpg                                     27641 ns        27641 ns        25719
vision_nms/eagle.jpg                                   29893 ns        29893 ns        23634
vision_nms/giraffe.jpg                                 27058 ns        27058 ns        26181
vision_nms/horses.jpg                                  28656 ns        28656 ns        24145
vision_nms/kite.jpg                                    29521 ns        29521 ns        23780
vision_nms/person.jpg                                  28229 ns        28229 ns        25179
faster_nms/dog.jpg                                    208364 ns       208364 ns         3387
faster_nms/eagle.jpg                                  206595 ns       206595 ns         3294
faster_nms/giraffe.jpg                                211329 ns       211330 ns         3330
faster_nms/horses.jpg                                 225208 ns       225209 ns         3154
faster_nms/kite.jpg                                   212902 ns       212902 ns         3133
faster_nms/person.jpg                                 212430 ns       212431 ns         3294
faster_nms_with_scores_threshold_sort/dog.jpg         125596 ns       126113 ns         5539
faster_nms_with_scores_threshold_sort/eagle.jpg       129874 ns       130408 ns         5412
faster_nms_with_scores_threshold_sort/giraffe.jpg     125974 ns       126493 ns         5401
faster_nms_with_scores_threshold_sort/horses.jpg      122550 ns       123064 ns         5514
faster_nms_with_scores_threshold_sort/kite.jpg        123377 ns       123879 ns         5592
faster_nms_with_scores_threshold_sort/person.jpg      123544 ns       124056 ns         5657
cv2_nms/dog.jpg                                         9001 ns         9001 ns        77577
cv2_nms/eagle.jpg                                       9055 ns         9055 ns        76856
cv2_nms/giraffe.jpg                                     8974 ns         8974 ns        77821
cv2_nms/horses.jpg                                      8855 ns         8855 ns        77020
cv2_nms/kite.jpg                                        9028 ns         9028 ns        74896
cv2_nms/person.jpg                                      8931 ns         8931 ns        78241
mmcv_nms/dog.jpg                                      172406 ns       172406 ns         3985
mmcv_nms/eagle.jpg                                    160593 ns       160593 ns         4390
mmcv_nms/giraffe.jpg                                  173412 ns       173412 ns         4107
mmcv_nms/horses.jpg                                   172767 ns       172767 ns         4070
mmcv_nms/kite.jpg                                     178548 ns       178548 ns         3899

```


### Yolov5 Results
```
----------------------------------------------------------------------------------------------
Benchmark                                                Time             CPU   Iterations
----------------------------------------------------------------------------------------------
nms_simd/dog.jpg                                       85472 ns        85472 ns         8053
nms_simd/eagle.jpg                                     84480 ns        84479 ns         8176
nms_simd/giraffe.jpg                                   84742 ns        84741 ns         8200
nms_simd/horses.jpg                                    87205 ns        87205 ns         8018
nms_simd/kite.jpg                                      87569 ns        87568 ns         7813
nms_simd/person.jpg                                    85620 ns        85620 ns         7937
tensorflow_nms/dog.jpg                                208716 ns       208716 ns         3225
tensorflow_nms/eagle.jpg                              206328 ns       206328 ns         3217
tensorflow_nms/giraffe.jpg                            209435 ns       209435 ns         3218
tensorflow_nms/horses.jpg                             269790 ns       269790 ns         2581
tensorflow_nms/kite.jpg                               235476 ns       235476 ns         3050
tensorflow_nms/person.jpg                             225714 ns       225714 ns         2946
vision_nms_with_score_threshold/dog.jpg               309261 ns       309691 ns         2344
vision_nms_with_score_threshold/eagle.jpg             268440 ns       268799 ns         2410
vision_nms_with_score_threshold/giraffe.jpg           272666 ns       273063 ns         2538
vision_nms_with_score_threshold/horses.jpg            279910 ns       280274 ns         2546
vision_nms_with_score_threshold/kite.jpg              283086 ns       283483 ns         2411
vision_nms_with_score_threshold/person.jpg            280230 ns       280614 ns         2575
vision_nms/dog.jpg                                1491354227 ns   1491354960 ns            1
vision_nms/eagle.jpg                              1372858286 ns   1372858727 ns            1
vision_nms/giraffe.jpg                            1460026741 ns   1460017250 ns            1
vision_nms/horses.jpg                             1377697945 ns   1377688976 ns            1
vision_nms/kite.jpg                               2007745266 ns   2007746124 ns            1
vision_nms/person.jpg                             1564868927 ns   1564868867 ns            1
faster_nms/dog.jpg                                  12692456 ns     12692506 ns           55
faster_nms/eagle.jpg                                13799845 ns     13799866 ns           55
faster_nms/giraffe.jpg                              13859332 ns     13859386 ns           52
faster_nms/horses.jpg                               12654918 ns     12654954 ns           54
faster_nms/kite.jpg                                 12135736 ns     12135601 ns           56
faster_nms/person.jpg                               12137779 ns     12137808 ns           58
faster_nms_with_scores_threshold_sort/dog.jpg        4863297 ns      4863883 ns          144
faster_nms_with_scores_threshold_sort/eagle.jpg      4868007 ns      4868626 ns          144
faster_nms_with_scores_threshold_sort/giraffe.jpg    4891262 ns      4891960 ns          143
faster_nms_with_scores_threshold_sort/horses.jpg     4959100 ns      4959899 ns          142
faster_nms_with_scores_threshold_sort/kite.jpg       4872549 ns      4873142 ns          144
faster_nms_with_scores_threshold_sort/person.jpg     5048616 ns      5049466 ns          142
cv2_nms/dog.jpg                                     71732855 ns     71732917 ns           10
cv2_nms/eagle.jpg                                   71560836 ns     71560054 ns           10
cv2_nms/giraffe.jpg                                 72858930 ns     72858055 ns           10
cv2_nms/horses.jpg                                  72767854 ns     72767087 ns           10
cv2_nms/kite.jpg                                    72154310 ns     72154181 ns            9
cv2_nms/person.jpg                                  72337914 ns     72337857 ns           10
mmcv_nms/dog.jpg                                      524275 ns       524275 ns         1297
mmcv_nms/eagle.jpg                                    515771 ns       515772 ns         1441
mmcv_nms/giraffe.jpg                                  549874 ns       549877 ns         1376
mmcv_nms/horses.jpg                                   510867 ns       510869 ns         1147
mmcv_nms/kite.jpg                                     547141 ns       547143 ns         1000
```

## How to Build

Currently there is no mechanism for `make install` for C++ libraries. If you are using CMake, then you can add this repo to your repo as submodule, then include this repo using `add_subdirectory`.

For building dependencies, you will only need cmake and a c++17 compatible compiler.

To install python package, you require [pybind11](https://github.com/pybind/pybind11). 

To build tests, you also need [googletest](https://github.com/google/googletest), [pcg-cpp](https://github.com/imneme/pcg-cpp) and [opencv](https://github.com/opencv/opencv). 

### Installing Python Package
Just run `python setup.py install`. To use vcpkg, you need to have `VCPKG_ROOT` environment variable to point to vcpkg installation root.

## Usage

### Python 

py_nms_simd.run(boxes, scores, score_threshold, nms_threshold)

#### Returns
A list that represents selected indices.

#### Arguments
- boxes: 2D numpy array with (N, 4) shape or 2D list that represents boxes
- scores: 1D numpy array with (N, ) shape or a list.
- score_threshold: A float value to threshold to filter boxes by score.
- nms_threshold: A float value to threshold boxes overlap.




	?Z'.??_@?Z'.??_@!?Z'.??_@	???q?#@???q?#@!???q?#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Z'.??_@Z???????A??xx??\@Y?3h??X(@*	d;?OO??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?r?4 R@!N+{)?X@)?r?4 R@1N+{)?X@:Preprocessing2F
Iterator::Model?%???{??!???-?׵?)?(?ޥ?1y"?X??:Preprocessing2P
Iterator::Model::PrefetchD????9??!Zj<q:???)D????9??1Zj<q:???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap>u?Rz R@!????X@)=??@fgq?1krG"&x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9???q?#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z???????Z???????!Z???????      ??!       "      ??!       *      ??!       2	??xx??\@??xx??\@!??xx??\@:      ??!       B      ??!       J	?3h??X(@?3h??X(@!?3h??X(@R      ??!       Z	?3h??X(@?3h??X(@!?3h??X(@JCPU_ONLYY???q?#@b 
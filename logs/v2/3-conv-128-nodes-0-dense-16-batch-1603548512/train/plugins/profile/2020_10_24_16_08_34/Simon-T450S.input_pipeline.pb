	1A?B?m@1A?B?m@!1A?B?m@	????????????!??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$1A?B?m@؜?gB??A?&P?"?m@Y??a?7???*	$???(?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???y?R@!??6+?X@)???y?R@1??6+?X@:Preprocessing2F
Iterator::Model?W?\T??!?aD ???)??6p???1??I????:Preprocessing2P
Iterator::Model::Prefetch%?YI+???!???fڧ?)%?YI+???1???fڧ?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapz?蹓R@!??????X@)??~m??o?1Η?.{u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	؜?gB??؜?gB??!؜?gB??      ??!       "      ??!       *      ??!       2	?&P?"?m@?&P?"?m@!?&P?"?m@:      ??!       B      ??!       J	??a?7?????a?7???!??a?7???R      ??!       Z	??a?7?????a?7???!??a?7???JCPU_ONLYY??????b 
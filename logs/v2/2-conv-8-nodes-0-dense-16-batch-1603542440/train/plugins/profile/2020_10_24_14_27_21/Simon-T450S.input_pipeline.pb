	??@?9wJ@??@?9wJ@!??@?9wJ@	/N?m[BC@/N?m[BC@!/N?m[BC@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??@?9wJ@??/????A?"??~
@@Y?#nkc4@*	???ƣ??@2P
Iterator::Model::Prefetch???`?P4@!???r?X@)???`?P4@1???r?X@:Preprocessing2F
Iterator::Model???`?Z4@!      Y@)??C p??1+m~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 38.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no90N?m[BC@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??/??????/????!??/????      ??!       "      ??!       *      ??!       2	?"??~
@@?"??~
@@!?"??~
@@:      ??!       B      ??!       J	?#nkc4@?#nkc4@!?#nkc4@R      ??!       Z	?#nkc4@?#nkc4@!?#nkc4@JCPU_ONLYY0N?m[BC@b 
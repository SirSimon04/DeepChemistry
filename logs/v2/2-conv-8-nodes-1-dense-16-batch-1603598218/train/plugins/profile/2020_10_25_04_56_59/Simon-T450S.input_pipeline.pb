	@?????H@@?????H@!@?????H@	??4?E@??4?E@!??4?E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$@?????H@?il????Al?˸?;@Yl>?+5@*	??|????@2P
Iterator::Model::Prefetch3??O5@!y)*???X@)3??O5@1y)*???X@:Preprocessing2F
Iterator::Modelv??2S"5@!      Y@)Y?8??m??1???k*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 43.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9??4?E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?il?????il????!?il????      ??!       "      ??!       *      ??!       2	l?˸?;@l?˸?;@!l?˸?;@:      ??!       B      ??!       J	l>?+5@l>?+5@!l>?+5@R      ??!       Z	l>?+5@l>?+5@!l>?+5@JCPU_ONLYY??4?E@b 
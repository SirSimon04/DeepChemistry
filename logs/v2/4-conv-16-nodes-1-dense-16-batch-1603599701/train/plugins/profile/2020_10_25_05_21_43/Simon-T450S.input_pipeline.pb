	??1>?TJ@??1>?TJ@!??1>?TJ@	wB?@?u6@wB?@?u6@!wB?@?u6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??1>?TJ@?)????A2????D@Y}x? #?'@*	??(\?
?@2P
Iterator::Model::Prefetch?%?`?'@!??!D?X@)?%?`?'@1??!D?X@:Preprocessing2F
Iterator::ModelF??j??'@!      Y@) 7?C??1???ƽw??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9wB?@?u6@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)?????)????!?)????      ??!       "      ??!       *      ??!       2	2????D@2????D@!2????D@:      ??!       B      ??!       J	}x? #?'@}x? #?'@!}x? #?'@R      ??!       Z	}x? #?'@}x? #?'@!}x? #?'@JCPU_ONLYYwB?@?u6@b 
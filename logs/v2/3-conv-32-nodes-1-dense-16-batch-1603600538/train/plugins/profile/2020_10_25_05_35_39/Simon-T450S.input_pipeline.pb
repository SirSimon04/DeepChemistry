	?GdR@?GdR@!?GdR@	ܬ77?@ܬ77?@!ܬ77?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?GdR@?u?ݑ1??A?d?P3dQ@Y?X??@*	?O??p??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator{JΉ=Q@!???|?W@){JΉ=Q@1???|?W@:Preprocessing2P
Iterator::Model::Prefetch????zI@!i?4@)????zI@1i?4@:Preprocessing2F
Iterator::Model*S?A?q@!?<?eV$@)?????*??1????!??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap|&??iQ@!7?????W@)?? n/f?1A?z?n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ܬ77?@#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?u?ݑ1???u?ݑ1??!?u?ݑ1??      ??!       "      ??!       *      ??!       2	?d?P3dQ@?d?P3dQ@!?d?P3dQ@:      ??!       B      ??!       J	?X??@?X??@!?X??@R      ??!       Z	?X??@?X??@!?X??@JCPU_ONLYYܬ77?@b 
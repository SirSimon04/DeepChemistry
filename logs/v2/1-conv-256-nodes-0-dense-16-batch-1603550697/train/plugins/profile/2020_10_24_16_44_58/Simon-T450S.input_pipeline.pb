	?#??/m@?#??/m@!?#??/m@	??????????????!???????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?#??/m@?Ϛi??A?tu?b%m@Y????????*	?rh?)??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator(??G?NQ@!dD???X@)(??G?NQ@1dD???X@:Preprocessing2F
Iterator::ModelD?|?F??!3?|?????)1???z??1@??W?ʗ?:Preprocessing2P
Iterator::Model::Prefetch&???o???!%??Q\??)&???o???1%??Q\??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?Cn?OQ@!i0ό-?X@)m7?7M?m?1?F??au?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Ϛi???Ϛi??!?Ϛi??      ??!       "      ??!       *      ??!       2	?tu?b%m@?tu?b%m@!?tu?b%m@:      ??!       B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????JCPU_ONLYY???????b 
	?Jm??@?Jm??@!?Jm??@	?(lu???(lu??!?(lu??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Jm??@???ם?3@A?]?????@Y?ȭI?@*	?x?&ŏA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?t???m@!?X?L??X@)?t???m@1?X?L??X@:Preprocessing2P
Iterator::Model::Prefetch??d?????!/???p???)??d?????1/???p???:Preprocessing2F
Iterator::Model????(y??!NVq!???)??.\???1m?'S???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapg)YNm@!S????X@)J???????1?P????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?(lu??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???ם?3@???ם?3@!???ם?3@      ??!       "      ??!       *      ??!       2	?]?????@?]?????@!?]?????@:      ??!       B      ??!       J	?ȭI?@?ȭI?@!?ȭI?@R      ??!       Z	?ȭI?@?ȭI?@!?ȭI?@JCPU_ONLYY?(lu??b 
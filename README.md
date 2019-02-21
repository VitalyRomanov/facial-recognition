Repository contains code for training model for unconstrained facial recognition. 

### Data

Model trained on preprocessed LFW dataset. `test_set.csv` contains the subsample of LFW that is used as the holdout test set. 

The preprocessing includes alignment and cropping, as described [here](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8?gi=676fb2d594e1).

### Architecture

Model uses pretrained CNN network. CNN weights are loaded as a ProtoBuf graph, and are not fine-tuned during the training. The script `loading_and_saving_weights.py` contains functions for exporting pretrained TensorFlow model into ProtoBuf format. The model definition and weights are borrowed from [TensorFlow Slim](https://github.com/tensorflow/models/tree/master/research/slim) repository. 

`main.py` creates three additional layers that process embeddings created by CNN network and extract facial features. The additional layers are trained using triplet loss. The final embeddings have dimensionality of 128.

Classification accuracy on the test set after 500 epochs is 73%. 
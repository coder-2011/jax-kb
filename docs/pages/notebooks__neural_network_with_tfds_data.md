- [](../index.html)
- [Resources and Advanced Guides](../advanced_guides.html)
- Training a simple neural network, with tensorflow/datasets data loading

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .ipynb](../_sources/notebooks/neural_network_with_tfds_data.ipynb "Download source file")
-  .pdf

# Training a simple neural network, with tensorflow/datasets data loading

## Contents

- [Hyperparameters](#hyperparameters)
- [Auto-batching predictions](#auto-batching-predictions)
- [Utility and loss functions](#utility-and-loss-functions)
- [Data loading with `tensorflow/datasets`](#data-loading-with-tensorflow-datasets)
- [Training loop](#training-loop)

**Copyright 2018 The JAX Authors.**

Licensed under the Apache License, Version 2.0 (the “License”);

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Training a simple neural network, with tensorflow/datasets data loading[\#](#training-a-simple-neural-network-with-tensorflow-datasets-data-loading "Link to this heading")

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/neural_network_with_tfds_data.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/neural_network_with_tfds_data.ipynb)

*Forked from* `neural_network_and_data_loading.ipynb`

![JAX](https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png)

Let’s combine everything we showed in the [quickstart](https://docs.jax.dev/en/latest/quickstart.html) to train a simple neural network. We will first specify and train a simple MLP on MNIST using JAX for the computation. We will use `tensorflow/datasets` data loading API to load images and labels (because it’s pretty great, and the world doesn’t need yet another data loading library :P).

Of course, you can use JAX with any API that is compatible with NumPy to make specifying the model a bit more plug-and-play. Here, just for explanatory purposes, we won’t use any neural network libraries or special APIs for building our model.

    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax import random

## Hyperparameters[\#](#hyperparameters "Link to this heading")

Let’s get a few bookkeeping items out of the way.

    # A helper function to randomly initialize weights and biases
    # for a dense neural network layer
    def random_layer_params(m, n, key, scale=1e-2):
      w_key, b_key = random.split(key)
      return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(sizes, key):
      keys = random.split(key, len(sizes))
      return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    layer_sizes = [784, 512, 512, 10]
    step_size = 0.01
    num_epochs = 10
    batch_size = 128
    n_targets = 10
    params = init_network_params(layer_sizes, random.key(0))

## Auto-batching predictions[\#](#auto-batching-predictions "Link to this heading")

Let us first define our prediction function. Note that we’re defining this for a *single* image example. We’re going to use JAX’s `vmap` function to automatically handle mini-batches, with no performance penalty.

    from jax.scipy.special import logsumexp

    def relu(x):
      return jnp.maximum(0, x)

    def predict(params, image):
      # per-example predictions
      activations = image
      for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

      final_w, final_b = params[-1]
      logits = jnp.dot(final_w, activations) + final_b
      return logits - logsumexp(logits)

Let’s check that our prediction function only works on single images.

    # This works on single examples
    random_flattened_image = random.normal(random.key(1), (28 * 28,))
    preds = predict(params, random_flattened_image)
    print(preds.shape)

    (10,)

    # Doesn't work with a batch
    random_flattened_images = random.normal(random.key(1), (10, 28 * 28))
    try:
      preds = predict(params, random_flattened_images)
    except TypeError:
      print('Invalid shapes!')

    Invalid shapes!

    # Let's upgrade it to handle batches using `vmap`

    # Make a batched version of the `predict` function
    batched_predict = vmap(predict, in_axes=(None, 0))

    # `batched_predict` has the same call signature as `predict`
    batched_preds = batched_predict(params, random_flattened_images)
    print(batched_preds.shape)

    (10, 10)

At this point, we have all the ingredients we need to define our neural network and train it. We’ve built an auto-batched version of `predict`, which we should be able to use in a loss function. We should be able to use `grad` to take the derivative of the loss with respect to the neural network parameters. Last, we should be able to use `jit` to speed up everything.

## Utility and loss functions[\#](#utility-and-loss-functions "Link to this heading")

    def one_hot(x, k, dtype=jnp.float32):
      """Create a one-hot encoding of x of size k."""
      return jnp.array(x[:, None] == jnp.arange(k), dtype)

    def accuracy(params, images, targets):
      target_class = jnp.argmax(targets, axis=1)
      predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
      return jnp.mean(predicted_class == target_class)

    def loss(params, images, targets):
      preds = batched_predict(params, images)
      return -jnp.mean(preds * targets)

    @jit
    def update(params, x, y):
      grads = grad(loss)(params, x, y)
      return [(w - step_size * dw, b - step_size * db)
              for (w, b), (dw, db) in zip(params, grads)]

## Data loading with `tensorflow/datasets`[\#](#data-loading-with-tensorflow-datasets "Link to this heading")

JAX is laser-focused on program transformations and accelerator-backed NumPy, so we don’t include data loading or munging in the JAX library. There are already a lot of great data loaders out there, so let’s just use them instead of reinventing anything. We’ll use the `tensorflow/datasets` data loader.

    import tensorflow as tf
    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type='GPU')

    import tensorflow_datasets as tfds

    data_dir = '/tmp/tfds'

    # Fetch full datasets for evaluation
    # tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
    # You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    num_labels = info.features['label'].num_classes
    h, w, c = info.features['image'].shape
    num_pixels = h * w * c

    # Full train set
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    train_labels = one_hot(train_labels, num_labels)

    # Full test set
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    test_labels = one_hot(test_labels, num_labels)

    print('Train:', train_images.shape, train_labels.shape)
    print('Test:', test_images.shape, test_labels.shape)

    Train: (60000, 784) (60000, 10)
    Test: (10000, 784) (10000, 10)

## Training loop[\#](#training-loop "Link to this heading")

    import time

    def get_train_batches():
      # as_supervised=True gives us the (image, label) as a tuple instead of a dict
      ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
      # You can build up an arbitrary tf.data input pipeline
      ds = ds.batch(batch_size).prefetch(1)
      # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
      return tfds.as_numpy(ds)

    for epoch in range(num_epochs):
      start_time = time.time()
      for x, y in get_train_batches():
        x = jnp.reshape(x, (len(x), num_pixels))
        y = one_hot(y, num_labels)
        params = update(params, x, y)
      epoch_time = time.time() - start_time

      train_acc = accuracy(params, train_images, train_labels)
      test_acc = accuracy(params, test_images, test_labels)
      print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
      print("Training set accuracy {}".format(train_acc))
      print("Test set accuracy {}".format(test_acc))

    Epoch 0 in 28.30 sec
    Training set accuracy 0.8400499820709229
    Test set accuracy 0.8469000458717346
    Epoch 1 in 14.74 sec
    Training set accuracy 0.8743667006492615
    Test set accuracy 0.8803000450134277
    Epoch 2 in 14.57 sec
    Training set accuracy 0.8901500105857849
    Test set accuracy 0.8957000374794006
    Epoch 3 in 14.36 sec
    Training set accuracy 0.8991333246231079
    Test set accuracy 0.903700053691864
    Epoch 4 in 14.20 sec
    Training set accuracy 0.9061833620071411
    Test set accuracy 0.9087000489234924
    Epoch 5 in 14.89 sec
    Training set accuracy 0.9113333225250244
    Test set accuracy 0.912600040435791
    Epoch 6 in 13.95 sec
    Training set accuracy 0.9156833291053772
    Test set accuracy 0.9176000356674194
    Epoch 7 in 13.32 sec
    Training set accuracy 0.9192000031471252
    Test set accuracy 0.9214000701904297
    Epoch 8 in 13.55 sec
    Training set accuracy 0.9222500324249268
    Test set accuracy 0.9241000413894653
    Epoch 9 in 13.40 sec
    Training set accuracy 0.9253666996955872
    Test set accuracy 0.9269000291824341

We’ve now used most of the JAX API: `grad` for derivatives, `jit` for speedups and `vmap` for auto-vectorization. We used NumPy to specify all of our computation, and borrowed the great data loaders from `tensorflow/datasets`, and ran the whole thing on the GPU.

[](../export/jax2tf.html "previous page")

previous

Interoperation with TensorFlow

[](Neural_Network_and_Data_Loading.html "next page")

next

Training a simple neural network, with PyTorch data loading

Contents

- [Hyperparameters](#hyperparameters)
- [Auto-batching predictions](#auto-batching-predictions)
- [Utility and loss functions](#utility-and-loss-functions)
- [Data loading with `tensorflow/datasets`](#data-loading-with-tensorflow-datasets)
- [Training loop](#training-loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\

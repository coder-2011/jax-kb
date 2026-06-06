- [](../index.html)
- [Resources and Advanced Guides](../advanced_guides.html)
- Training a simple neural network, with PyTorch data loading

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .ipynb](../_sources/notebooks/Neural_Network_and_Data_Loading.ipynb "Download source file")
-  .pdf

# Training a simple neural network, with PyTorch data loading

## Contents

- [Hyperparameters](#hyperparameters)
- [Auto-batching predictions](#auto-batching-predictions)
- [Utility and loss functions](#utility-and-loss-functions)
- [Data loading with PyTorch](#data-loading-with-pytorch)
- [Training loop](#training-loop)

# Training a simple neural network, with PyTorch data loading[\#](#training-a-simple-neural-network-with-pytorch-data-loading "Link to this heading")

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb)

**Copyright 2018 The JAX Authors.**

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

![JAX](https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png)

Let’s combine everything we showed in the [quickstart](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/quickstart.html) to train a simple neural network. We will first specify and train a simple MLP on MNIST using JAX for the computation. We will use PyTorch’s data loading API to load images and labels (because it’s pretty great, and the world doesn’t need yet another data loading library).

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
    num_epochs = 8
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

## Data loading with PyTorch[\#](#data-loading-with-pytorch "Link to this heading")

JAX is laser-focused on program transformations and accelerator-backed NumPy, so we don’t include data loading or munging in the JAX library. There are already a lot of great data loaders out there, so let’s just use them instead of reinventing anything. We’ll grab PyTorch’s data loader, and make a tiny shim to make it work with NumPy arrays.

    !pip install torch torchvision

    Requirement already satisfied: torch in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (2.4.1)
    Requirement already satisfied: torchvision in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (0.19.1)
    Requirement already satisfied: filelock in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (3.16.0)
    Requirement already satisfied: typing-extensions>=4.8.0 in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (1.13.2)
    Requirement already satisfied: networkx in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (2024.9.0)
    Requirement already satisfied: setuptools in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torch) (73.0.1)
    Requirement already satisfied: numpy in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torchvision) (1.26.4)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from torchvision) (10.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages (from sympy->torch) (1.3.0)

    /home/m/.opt/miniforge3/envs/jax/lib/python3.12/pty.py:95: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      pid, fd = os.forkpty()

    import numpy as np
    from jax.tree_util import tree_map
    from torch.utils.data import DataLoader, default_collate
    from torchvision.datasets import MNIST

    def numpy_collate(batch):
      """
      Collate function specifies how to combine a list of data samples into a batch.
      default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
      """
      return tree_map(np.asarray, default_collate(batch))

    def flatten_and_cast(pic):
      """Convert PIL image to flat (1-dimensional) numpy array."""
      return np.ravel(np.array(pic, dtype=jnp.float32))

    # Define our dataset, using torch datasets
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)
    # Create pytorch data loader with custom collate function
    training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to /tmp/mnist/MNIST/raw/train-images-idx3-ubyte.gz
    Extracting /tmp/mnist/MNIST/raw/train-images-idx3-ubyte.gz to /tmp/mnist/MNIST/raw

    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to /tmp/mnist/MNIST/raw/train-labels-idx1-ubyte.gz
    Extracting /tmp/mnist/MNIST/raw/train-labels-idx1-ubyte.gz to /tmp/mnist/MNIST/raw

    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to /tmp/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz
    Extracting /tmp/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz to /tmp/mnist/MNIST/raw

    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to /tmp/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz
    Extracting /tmp/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz to /tmp/mnist/MNIST/raw

    100.0%
    100.0%
    100.0%
    100.0%

    # Get the full train dataset (for checking accuracy while training)
    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

    # Get full test dataset
    mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

    /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages/torchvision/datasets/mnist.py:76: UserWarning: train_data has been renamed data
      warnings.warn("train_data has been renamed data")
    /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages/torchvision/datasets/mnist.py:66: UserWarning: train_labels has been renamed targets
      warnings.warn("train_labels has been renamed targets")
    /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages/torchvision/datasets/mnist.py:81: UserWarning: test_data has been renamed data
      warnings.warn("test_data has been renamed data")
    /home/m/.opt/miniforge3/envs/jax/lib/python3.12/site-packages/torchvision/datasets/mnist.py:71: UserWarning: test_labels has been renamed targets
      warnings.warn("test_labels has been renamed targets")

## Training loop[\#](#training-loop "Link to this heading")

    import time

    for epoch in range(num_epochs):
      start_time = time.time()
      for x, y in training_generator:
        y = one_hot(y, n_targets)
        params = update(params, x, y)
      epoch_time = time.time() - start_time

      train_acc = accuracy(params, train_images, train_labels)
      test_acc = accuracy(params, test_images, test_labels)
      print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
      print("Training set accuracy {}".format(train_acc))
      print("Test set accuracy {}".format(test_acc))

    Epoch 0 in 5.53 sec
    Training set accuracy 0.9156666994094849
    Test set accuracy 0.9199000000953674
    Epoch 1 in 1.13 sec
    Training set accuracy 0.9370499849319458
    Test set accuracy 0.9383999705314636
    Epoch 2 in 1.12 sec
    Training set accuracy 0.9490833282470703
    Test set accuracy 0.9467999935150146
    Epoch 3 in 1.21 sec
    Training set accuracy 0.9568833708763123
    Test set accuracy 0.9532999992370605
    Epoch 4 in 1.17 sec
    Training set accuracy 0.9631666541099548
    Test set accuracy 0.9574999809265137
    Epoch 5 in 1.17 sec
    Training set accuracy 0.9675000309944153
    Test set accuracy 0.9615999460220337
    Epoch 6 in 1.11 sec
    Training set accuracy 0.9709500074386597
    Test set accuracy 0.9652999639511108
    Epoch 7 in 1.17 sec
    Training set accuracy 0.9736999869346619
    Test set accuracy 0.967199981212616

We’ve now used the whole of the JAX API: `grad` for derivatives, `jit` for speedups and `vmap` for auto-vectorization. We used NumPy to specify all of our computation, and borrowed the great data loaders from PyTorch, and ran the whole thing on the GPU.

[](neural_network_with_tfds_data.html "previous page")

previous

Training a simple neural network, with tensorflow/datasets data loading

[](vmapped_log_probs.html "next page")

next

Autobatching for Bayesian inference

Contents

- [Hyperparameters](#hyperparameters)
- [Auto-batching predictions](#auto-batching-predictions)
- [Utility and loss functions](#utility-and-loss-functions)
- [Data loading with PyTorch](#data-loading-with-pytorch)
- [Training loop](#training-loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\

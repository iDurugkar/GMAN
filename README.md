The repository contains code to experiment on Generative Adversarial Nets
with multiple Discriminators.

An example is given below:

```
$ python GMAN.py --dataset mnist --num_disc 1 --lam 0. --path testing_dataset

$ python GMAN.py --dataset Data/my_images --num_disc 1 --lam 0. --path testing_dataset
```

<p>There are three standard datasets you can run on:
MNIST, CIFAR-10 and CelebA.</p>
<p>You can run on these 3 datasets just by mentioning them by name as above.</p>
<p>To download these, you can run the `download.py` file.
It will download the dataset in the `./Data` directory. The instructions
to use are:</p>

```
$ python download.py mnist
$ python download.py celebA
$ python download.py cifar
```

You can then run
```
$ python GMAN.py
```
The various arguments that can be passed can be seen at the bottom of the code.

<h2>Alternate Dataset</h2>
<p>You can load your custom dataset as well. The dataset should be image files or
a `.npy` array with shape `(dataset_size, 32, 32, num_channels)`.
Set the flag `--dataset` and give the path to the `.npy` array or the directory
of images to load.</p>

<p>You should also set the `--path` parameter to the path where you want to 
save the results. It won't work otherwise.</p>


The code automatically normalizes the data to `[-1, 1]`.

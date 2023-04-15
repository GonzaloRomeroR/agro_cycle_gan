.. agro-cycle-gan documentation master file, created by
   sphinx-quickstart on Sun Feb  6 10:01:17 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. In order to update the documentation in readthedocs you should make the modification in this file
   then log in into read the docs and rebuild the documentation

Agro Cycle Gan
==========================================

Agro Cycle Gan is a platform to create syntetic image datasets using CycleGAN. It transforms images from a domain A with some features to a domain B with different features (for example horses to horses, day to night, etc). It uses CycleGAN, which allows to use unsupervised learning to transform images between two domains. 


Images
------------

In order to train a generator to be able to translate from one domain A to a domain B, add your images in the *images/* folder. The folder structure must contain must contain the following folders:

* images/*name_of_dataset*/train_A/A/
* images/*name_of_dataset*/test_A/A/
* images/*name_of_dataset*/train_B/B/
* images/*name_of_dataset*/test_B/B/

Where train_A contains the images to train from the domain A and train_B contains images to train from the domain B. The same happens for the testing dataset. The datasets must have the previous format in order to be used by the platform.

If your dataset is divided in two folder containing the images in two domains, it is possible to use the *scripts/split_dataset.py* script to obtain the required format.

``python scripts/split_datasets.py ./images/image_folder_A ./images/image_folder_B imagesA2imagesB``

The images dataset is going to be stored in the images/ folder. Use the the format *images_a2images_b* as folder name.


Installation
------------

Clone the repository:

``git clone https://github.com/GonzaloRomeroR/agro_cycle_gan.git``


Install the required packages: ``pip install environments/requirements.txt``

Trainer
-------

To train the models use:

``python -u train.py horse2zebra --batch_size 10 --num_epochs 20 --image_resize 64 64 --plot_image_epoch``


Where horse2zebra should be replaced by the name of your dataset. Several options can be used to train the generator:


* *--download_dataset* (boolean): in case that the dataset is not found in the images/ folder then the platform will try to download it (only for predefined dataset in the platform)
* *--image_resize* (list of ints): resizing of the images been used (e.g --image_resize 256 256)
* *--tensorboard* (boolean): generate tensorboard files
* *--store_modes* (boolean): this will store the generator and discriminator in a folder with the datetime once the training is finished.
* *--load_models* (boolean): load models trained in previous training processes
* *--batch_size* (int): batch size to use during training (default to 5)
* *--num_epochs* (int): number of epochs to use during training (default to 1)
* *--metrics* (boolean): if true, the platform will calculate the training metrics every epochs
* *--plot_image_epoch* (boolean): if true, the platform will display an example of generated image every epochs
* *--generator* (string): name of the generator to use
* *--discriminator* (string): name of the disciriminator to use
* *--comments* (string): comments to add in results artifacts

Right now the following generators are available:

* *cyclegan*: base cyclegan generator
* *mixer*: mixer gan generator
* *resnet*: cyclegan with different architecture

Right now the following disciriminators are available:

* *basic*: basic cyclegan discriminator
* *nlayer*: cyclegan discriminator with different architecture


Generation
----------

In order to generate images run:

``python generate.py ./images/horse2zebra/test_A/A ./images_gen/horse2zebra/ --generator_name horse2zebra --dest_domain B``


Where *./images/horse2zebra/test_A/A* is the folder containing the images to transform, *./images_gen/horse2zebra/* is the destination folder where the transformed images will be stored. The generator name must be equal to the dataset name. It is possible to select the destination domain with *--dest_domain* (it can be A or B).

Examples:
--------

``python -u train.py horse2zebra --batch_size 10 --image_resize 64 64 --tensorboard --store_modes --batch_size 10 --num_epochs 20 --metrics --plot_image_epoch --generator cyclegan --discriminator basic --comments "Example run"``

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

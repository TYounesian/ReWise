# ReWise #

This is the official implementation of "ReWise: A Relation-Wise Sampling Framework for Relational Graph Convolutional Networks". This paper is under submission for SEMANTICS 2024. 

## Instructions ##
## 1. Install dependencies ##
First, create an environment using the environment file provided, then activate it:
```
conda env create -f rewise-env.yml
conda activate rewise
```

Then install ``kgbench`` by downloading or cloning the kgbench repository [https://github.com/pbloem/kgbench-loader] and follow the installation steps or in the root 
directory, run ``pip install .``

## 2. Run an experiment ##
Run the following to train and test the RGCN with ReWise-LDRN for ``amplus``:
```
python main.py
```

For other datasets, hyperparameters, samplers, and settings specify the corresponding inputs. For example, to run ReWise-LDRN for ``dmgfull`` with multimodal features and sample size 64, run the following:
```
python main.py --data_name='dmgfull' --modality='all' --samp0=64 
```

## 3. Try different samplers ##
The input sampler, accepts the following options: ``LDRN``, ``LDRE``, ``LDUN``, ``IDRN``, ``IDUN``, ``IARN``, ``IAUN``, and ``full-mini-batch``. Set the ``batch_size`` to -1 to train with the full-batch setting. 

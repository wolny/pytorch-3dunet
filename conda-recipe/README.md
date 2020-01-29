## Conda build
In order to create conda package:
1. Run: `conda build conda-recipe/ -c conda-forge`
2. Install locally with `conda install --use-local pytorch-3dunet -c conda-forge` (optional)
3. Upload to conda cloud: `anaconda upload /home/adrian/miniconda3/conda-bld/linux-64/pytorch-3dunet-<version>-py37_0.tar.bz2`
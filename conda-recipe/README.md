## Conda build
In order to create conda package:
1. Run: `conda build conda-recipe/ -c conda-forge`
2. Install locally with `conda install --use-local pytorch-3dunet -c conda-forge` (optional)
3. Upload to conda cloud: `anaconda upload /home/adrian/miniconda3/conda-bld/linux-64/pytorch-3dunet-<version>-py37_0.tar.bz2`

## Release new version
1. Make sure that `bumpversion` is installed in your conda env
2. Checkout master branch
3. Run `bumpversion patch` (or `major` or `minor`)
4. Run `git push && git push --tags` (trigger Travis build) 
5. The rest is going to be made by Travis (i.e. conda build + upload)
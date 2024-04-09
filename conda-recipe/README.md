## Conda build
In order to create conda package:
1. Run: `conda build -c pytorch -c conda-forge conda-recipe`
2. Install locally with `conda install -c pytorch -c conda-forge --use-local pytorch-3dunet` (optional)
3. Upload to conda cloud: `anaconda upload /home/adrian/miniconda3/conda-bld/linux-64/pytorch-3dunet-<version>-py37_0.tar.bz2`

## Release new version on `awolny` channel
1. Make sure that `bumpversion` is installed in your conda env
2. Checkout master branch
3. Run `bumpversion patch` (or `major` or `minor`) - this will bump the version in `.bumpversion.cfg` and `__version__.py` add create a new tag
4. Run `git push && git push --tags` (trigger github actions) 
5.  conda build + upload is done by github actions
6. 
## Release new version on `conda-forge` channel
1. Make a new release on GitHub 
2. (Optional) Make sure that the new release version is in sync with the version in `.bumpversion.cfg` and `__version__.py` (see above)
3. Generate the checksums for the new release using: `curl -sL https://github.com/wolny/pytorch-3dunet/archive/refs/tags/VERSION.tar.gz | openssl sha256`. Replace `VERSION` with the new release version
4. Fork the `conda-forge` feedstock  repository (https://github.com/conda-forge/pytorch-3dunet-feedstock)
5. Clone the forked repository and create a new PR with the following changes:
    - Update the `version` in `recipe/meta.yaml` to the new release version
    - Update the `sha256` in `recipe/meta.yaml` to the new checksum
6. Wait for the checks to pass. Once the PR is merged, the new version will be available on the `conda-forge` channel
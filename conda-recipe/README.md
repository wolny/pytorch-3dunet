## Conda build
In order to create conda package:
1. Run: `conda build conda-recipe/meta.yaml -c conda-forge`
2. Install locally with `conda install --use-local pytorch-3dunet -c conda-forge`
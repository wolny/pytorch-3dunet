import argparse

from pytorch3dunet.unet3d.config import override_config


class TestConfig:
    def test_override_config(self, test_config):
        args = argparse.Namespace(
            config=None,
            model_path="new_model.pth",
            **{
                "loaders.output_dir": "new_output_dir",
                "loaders.test.file_paths": ["file1.h5", "file2.h5"],
                "loaders.test.slice_builder.patch_shape": [64, 64, 64],
                "loaders.test.slice_builder.stride_shape": [32, 32, 32],
            },
        )

        override_config(args, test_config)

        assert test_config["model_path"] == "new_model.pth"
        assert test_config["loaders"]["output_dir"] == "new_output_dir"
        assert test_config["loaders"]["test"]["file_paths"] == ["file1.h5", "file2.h5"]
        assert test_config["loaders"]["test"]["slice_builder"]["patch_shape"] == [64, 64, 64]
        assert test_config["loaders"]["test"]["slice_builder"]["stride_shape"] == [32, 32, 32]

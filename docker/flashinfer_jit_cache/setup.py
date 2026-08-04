from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class BinaryWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        return "py3", "none", "linux_x86_64"


setup(cmdclass={"bdist_wheel": BinaryWheel})

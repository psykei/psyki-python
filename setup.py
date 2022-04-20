from setuptools import setup, find_packages
import pathlib
import subprocess
import distutils.cmd

# current directory

here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


def format_git_describe_version(version):
    if '-' in version:
        splitted = version.split('-')
        tag = splitted[0]
        index = f"dev{splitted[1]}"
        return f"{tag}.{index}"
    else:
        return version


def get_version_from_git():
    try:
        process = subprocess.run(["git", "describe"], cwd=str(here), check=True, capture_output=True)
        version = process.stdout.decode('utf-8').strip()
        version = format_git_describe_version(version)
        with version_file.open('w') as f:
            f.write(version)
        return version
    except subprocess.CalledProcessError:
        if version_file.exists():
            return version_file.read_text().strip()
        else:
            return '0.1.0'


version = get_version_from_git()


print(f"Detected version {version} from git describe")


class GetVersionCommand(distutils.cmd.Command):
    """A custom command to get the current project version inferred from git describe."""

    description = 'gets the project version from git describe'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(version)


class GenerateAntlr4Parser(distutils.cmd.Command):
    """A custom command to generate an Antlr4 parser class for a given grammar."""

    description = 'generate the Antlr4 parser for a given grammar'
    user_options = [('file=', 'f', 'grammar file name')]

    def initialize_options(self):
        from psyki.resources import PATH
        self.file = str(PATH / 'Datalog.g4')

    def finalize_options(self):
        pass

    def run(self):
        import re
        from os import system, popen
        antlr4_version = re.split(r'=', popen('cat requirements.txt | grep antlr4').read())[1][:-1]
        system('wget https://www.antlr.org/download/antlr-' + antlr4_version + '-complete.jar')
        system('export CLASSPATH="./antlr-' + antlr4_version + '-complete.jar:$CLASSPATH"')
        system('java -jar ./antlr-' + antlr4_version + '-complete.jar -Dlanguage=Python3 ' + self.file + ' -visitor -o psyki/resources/dist')
        system('rm ./antlr-' + antlr4_version + '-complete.jar')


setup(
    name='psyki',  # Required
    version=version,
    description='Python-based implementation of PSyKI, i.e. a Platform for Symbolic Knowledge Injection',
    license='Apache 2.0 License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/psykei/psyki-python',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Prolog'
    ],
    keywords='symbolic knowledge injection, ski, symbolic ai',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'antlr4-python3-runtime~=4.9.3',
        'tensorflow>=2.7.0',
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
        # 'skl2onnx~=1.10.0',
        # 'onnxruntime~=1.9.0'
    ],  # Optional
    zip_safe = False,
    platforms = "Independant",
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/psykei/psyki-python/issues',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/psykei/psyki-python',
    },
    cmdclass={
        'get_project_version': GetVersionCommand,
        'generate_antlr4_parser': GenerateAntlr4Parser,
    },
)
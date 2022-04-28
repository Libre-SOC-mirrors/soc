from setuptools import setup, find_packages
import sys
import os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()

version = '0.0.1'

# the only reason this is added is because it's become a part of python 3.9.
# the project standard is python 3.7 however in future that will be updated.
# for now, cached_property is RELUCTANTLY added but a *copy* is added so
# that the generation of HDL is not critically dependent on random crap
# off the internet. you're spending USD 16 *MILLION* on masks, you better
# be absolutely paranoid-level certain you know where every piece of the
# chain creating the HDL comes from.
cprop = "git+https://git.libre-soc.org/git/cached-property.git@1.5.2" \
        "#egg=cached-property-1.5.2"

# using pip3 for ongoing development is a royal pain.  seriously not
# recommended.  therefore a number of these dependencies have been
# commented out.  *they are still required* - they will need installing
# manually.

# XXX UNDER NO CIRCUMSTANCES ADD ARBITRARY DEPENDENCIES HERE. XXX
# as this is HDL, not software, every dependency added is
# a serious maintenance and reproducible-build problem.
# dropping USD 16 million on 7nm Mask Charges when the
# HDL can be compromised - accidentally or deliberately -
# by pip3 going out and randomly downloading complete
# shite is not going to do anyone any favours.

# TODO: make *all* of these be from libre-soc git repo only
# (which means updating the nmigen-soc one to mirror gitlab)

install_requires = [
    #    'sfpy',    # needs manual patching
    'libresoc-ieee754fpu',   # uploaded (successfully, whew) to pip
    'libresoc-openpower-isa',  # uploaded (successfully, whew) to pip
    # 'nmigen-soc', # install manually from git.libre-soc.org

    # git url needed for having `pip3 install -e .` install from libre-soc git
    "cached-property@"+cprop,
]

# git url needed for having `setup.py develop` install from libre-soc git
dependency_links = [
    cprop,
]

test_requires = [
    'nose',
    # install pia from https://salsa.debian.org/Kazan-team/power-instruction-analyzer
    'power-instruction-analyzer'
]

setup(
    name='libresoc',
    version=version,
    description="A nmigen-based OpenPOWER multi-issue Hybrid 3D CPU-VPU-GPU",
    long_description=README + '\n\n' + NEWS,
    long_description_content_type='text/markdown',
    classifiers=[
        "Topic :: Software Development",
        "License :: OSI Approved :: " \
            "GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords='nmigen ieee754 libre-soc soc',
    author='Luke Kenneth Casson Leighton',
    author_email='lkcl@libre-soc.org',
    url='http://git.libre-soc.org/?p=soc',
    license='LGPLv3+',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    dependency_links=dependency_links,
    tests_require=test_requires,
    test_suite='nose.collector',
)

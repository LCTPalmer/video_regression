from setuptools import setup

setup(name='multichannel',
      version='0.0.1',
      description='multichannel kernel support on top of sklearn models',
      author='Luke Palmer',
      author_email='lctpalmer@gmail.com',
      license='MIT',
      packages=['multichannel'],
      install_requires=[
          'scikit-learn',
          'numpy',
          'theano'
          ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

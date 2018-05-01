Intro Note
==========
* This is probably an overkill but at least it shows some (hopefully good) use of github / python's virtual environments / integration and unitary tests etc.

* Some choices were to be made, e.g. what activation function to use in the dense layer etc. Those choices do not affect the Tasks results, however.

* I commented a bit the functions so that some documentation can be generated easily (pydoc etc.)

* Tasks 1 and 3 are covered in the main.py file, in particular, my_rot.get_dense_output()     (see below)

* Task 2 is covered in the integration test (see below)

Installation
============

Once clone, 

* The best way is then to create a virtual env from which to run the code and tests: 
	* if not installed, install virtualenv:
		
			pip install virtualenv
	* build the virtual env using the requirements.txt file from the repo:
			
			cd guillaumeconvnets/
			virtualenv guillaume_env
			guillaume_env/bin/pip install -r requirements.txt

Note: this also installs the 'guillaumeml' package with pip, such that this package becomes available from anywhere on your system (see below)

Tests
=====

I designed a set of 2 integration tests (unitary tests can be added later). The code coverage is not high for time constraint reasons but this is something that should be taken care of in a real-life project.

Tests must be run from the root of the repo clone. 

Make sure your python environment points to the 'guillaume_env' built earlier, or any other proper environment:

	source ./guillaume_env/bin/activate
	
The full tests (unitary if any + integration) can be run using the following pythonic command (using pytest under the hood):

	python -m guillaumeml.tests.test_integration
	
Note a @timer_function wrapper decorates the integration tests, mostly to make sure the tests don't take too long.


Usage
=====

The default usage is a call to main.py from the repo's root dir.

Make sure your python environment points to the 'guillaume_env' built earlier, or any other proper environnement:

		source ./guillaume_env/bin/activate

This file can take arguments as inputs as can be seen by calling:
	
	python main.py --help
	
Examples: 
* To run an example: 

		python main.py --rotation_group 'dihedral'

* Another example with custom angles:
		
		python main.py --rotation_group 'custom' --angles 30 60 90 123 180 -89 364 720 
  
Using the guillaumeml module from anywhere in your system
=========================================================

Since guillaumeml is also pip-installed (see last line of requirements.txt and setup.py file in repo), it can be 'used' from any shell that has some python interpreter available (make sure guillaume_env is used):

	python  # use guillaume_env/bin/python or source the activate script first as above
	from guillaumefibo.lib.rotconv2d import RotConv2D
 
 	X = tf.reshape(tf.constant([[[0., 1, 0], [0., 1, 0], [0., 1, 0.]]]),
                   shape=[1, 3, 3, 1])  # last digit is # of channels
	K = tf.reshape(tf.constant([[[0., 0, 0], [1., 1, 1], [0., 0, 0]]]),
                   shape=[3, 3, 1, 1])  # last digit is # of output channels.. 2nd to last is # of input channels

	my_rot = RotConv2D(x_input=X, kernel=K, rotation_type=FLAGS.rotation_group)
	
	my_rot.initialize_all_filters()
	
	dense_output = my_rot.get_dense_output()  # THIS IS THE EXPECTED OUTPUT FOR TASK 1 AND TASK 3

	print 'Dense output: ', dense_output.eval(session=sess)



Potential Improvements on the DevOps side
================================
* One next step would be to use docker: 1) build a Dockerfile (from an ubuntu image + RUN git clone and pip install etc.) and 2) setup a DockerHub account that can be updated everytime something is pushed to this repo. 

* CircleCI can be used to do the automatic push to the DockerHub repo


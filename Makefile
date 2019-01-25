
PACKAGE = strudel.utils
TESTROOT = stutils

.PHONY: test
test:
	python -m doctest $(TESTROOT)/*.py
	python -m unittest test

.PHONY: publish
publish:
	$(MAKE) clean
	$(MAKE) test
	python setup.py sdist bdist_wheel
	twine upload dist/*
	$(MAKE) clean

.PHONY: clean
clean:
	rm -rf $(PACKAGE).egg-info dist build docs/build
	find -name "*.pyo" -delete
	find -name "*.pyc" -delete
	find -name __pycache__ -delete

.PHONY: html
html:
	sphinx-build -M html "docs" "docs/build"

.PHONY: install
install:
	pip install -r requirements.txt
	sudo apt-get install yajl-tools

.PHONY: install_dev
install_dev:
	$(MAKE) install
	pip install typing requests sphinx sphinx-autobuild
	pip install python-semantic-release==3.11.2

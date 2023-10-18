SHELL = bash

default: test

test:
	$(foreach T,$(wildcard *.py),python3 -m doctest $T;)

tinystories:
	python3 app.py tinystories

tinystories_mini:
	python3 app.py tinystories --model_size=260K
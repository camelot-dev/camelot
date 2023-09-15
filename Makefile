build:
	flit build

upload:
	make clean
	flit publish

clean:
	pyclean .
	rm -rf tests/__pycache__ camelot/__pycache__ dist

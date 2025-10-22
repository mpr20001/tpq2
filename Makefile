requirements:
	pip3 install -r ./requirements.txt

strata-test: build-deps
	pip3 install -r ./requirements.txt
	pytest --cov=query_predictor --cov-report=xml:coverage-reports/coverage.xml --cov-report=html:coverage-reports/coverage.html

build-deps:
	dnf -y install gcc gcc-c++ make python3-devel libgomp

test:
	pytest tests/
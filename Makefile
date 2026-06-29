.PHONY: configure build test cpp-test python-test baseline train-sync train-async serve load clean

configure:
	cmake -S . -B build

build: configure
	cmake --build build

cpp-test: build
	ctest --test-dir build --output-on-failure

python-test:
	python3 -m pytest -q

test: cpp-test python-test

baseline:
	python3 -m worker.train_worker --baseline --epochs 80 --output build/baseline.weights

train-sync: build
	python3 scripts/train_distributed.py --mode sync --workers 2 --epochs 80 --output build/model.weights

train-async: build
	python3 scripts/train_distributed.py --mode async --workers 4 --epochs 80 --output build/model.weights

serve: build
	build/inference_server --weights build/model.weights --port 50052 --max-batch-size 16 --max-delay-ms 5

load:
	python3 client/load_gen.py --port 50052 --requests 200 --concurrency 32

clean:
	cmake --build build --target clean || true

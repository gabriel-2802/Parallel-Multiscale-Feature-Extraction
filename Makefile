DIRS := \
	mpi \
	openmp \
	cuda \
	mpi_cuda \
	pthreads \
	serial \
	mpi_openmp \
	pthreads_openmp

.PHONY: all
all: build

.PHONY: build
build:
	@set -e; \
	for d in $(DIRS); do \
		echo Building $$d"; \
		$(MAKE) -C $$d build; \
	done

.PHONY: run
run:
	@set -e; \
	for d in $(DIRS); do \
		echo "Running $$d"; \
		$(MAKE) -C $$d run; \
	done

.PHONY: clean
clean:
	@set -e; \
	for d in $(DIRS); do \
		echo "Cleaning $$d"; \
		$(MAKE) -C $$d clean; \
	done

.PHONY: $(DIRS)

$(DIRS):
	$(MAKE) -C $@ build

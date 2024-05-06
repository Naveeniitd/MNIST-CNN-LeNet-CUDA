# Compiler
NVCC := nvcc -std=c++11

# Load modules
LOAD_CUDA := module load compiler/cuda/10.2/compilervars
LOAD_PYTHON := module load compiler/python/3.6.0/ucs4/gnu/447
LOAD_OPENCV := module load pythonpackages/3.6.0/opencv/3.4.1/gpu

# Directories
SRCDIR := src
BINDIR := .

# Targets
TARGETS := subtask1 subtask2 subtask3 subtask4

# Rules
all: $(addprefix $(BINDIR)/,$(TARGETS))
	$(LOAD_PYTHON) && $(LOAD_OPENCV) && python3 $/preprocessing.py

$(BINDIR)/%: $(SRCDIR)/assignment2_%.cu
	$(NVCC) $< -o $@

clean:
	rm -f $(addprefix $(BINDIR)/,$(TARGETS))

CC = gcc
CCFLAGS = -Wall -Wextra -Iinclude -I/opt/cuda/include
NV = nvcc
NVFLAGS = -Iinclude -I/opt/cuda/include

SRC_DIR = src
OBJ_DIR = obj
LSF_DIR = lsf

SRC_C = $(wildcard $(SRC_DIR)/*.c)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_C)) \
			$(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_CU))

LSF = $(wildcard $(LSF_DIR)/*.lsf)

TARGET = program

all: $(TARGET)

serial: CCFLAGS += -DSERIAL
serial: NVFLAGS += -DSERIAL
serial: $(TARGET)
	$(CC) $(CCFLAGS) -o "$(TARGET)_serial" $(patsubst $(OBJ_DIR)/%.o, $(SRC_DIR)/%.c, $(SRC_C))

parallel: $(TARGET)
	mv $(TARGET) "$(TARGET)_parallel"

$(TARGET): $(OBJ)
	$(NV) $(NVFLAGS) -o $@ $^ -L/opt/cuda/lib -lcudart -lcuda

DEP := $(patsubst $(OBJ_DIR)/%.o, $(OBJ_DIR)/%.d, $(OBJ))
-include $(DEP)
DEPFLAGS = -MMD -MF $(@:.o=.d)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)
	$(CC) $(CCFLAGS) -c $< -o $@ $(DEPFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(OBJ_DIR)
	$(NV) $(NVFLAGS) -c $< -o $@ $(DEPFLAGS)

clean:
	rm -f $(OBJ) $(DEP)

bsubload: $(TARGET)
	for lsf_script in $(LSF) ; do \
		bsub < $$lsf_script ; \
	done

lsf:
	echo $(KBLOCKS) $(KTHREADS) $(TYPE) $(ARRAY_SIZE) $(CYCLES)
	mkdir -p lsf

	$(eval JOB_NAME := "lab4_4_$(TYPE)_$(KBLOCKS)_$(KTHREADS)")
	$(eval PROJECT_NAME := "mothm_lab4_4")
	$(eval LOG_FILE := "$(JOB_NAME).log")

	echo -e "#!/bin/bash\nmkdir -p logs err\n\n#BSUB -J $(JOB_NAME)\n#BSUB -P $(PROJECT_NAME)\n#BSUB -W 08:00\n#BSUB -n $(THREADS)\n#BSUB -oo logs/$(LOG_FILE)\n#BSUB -eo err/$(LOG_FILE)\n\nexport ARRAY_SIZE=$(ARRAY_SIZE)\nexport CYCLES=$(CYCLES)\nexport KBLOCKS=$(KBLOCKS)\nexport KTHREADS=$(KTHREADS)\n\nmodule load cuda/11.4\n{ time ./program_$(TYPE) ; } 2> logs/$(JOB_NAME).time" > "./lsf/pr$(KBLOCKS)_$(KTHREADS)_$(TYPE).lsf"

	chmod +x ./lsf/pr$(KBLOCKS)_$(KTHREADS)_$(TYPE).lsf

.PHONY: lsf

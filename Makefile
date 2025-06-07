CC = g++
CCFLAGS = -Wall -Wextra -Iinclude -I/opt/cuda/include
NV = nvcc
NVFLAGS = -Iinclude -I/opt/cuda/include

SRC_DIR = src
OBJ_DIR = obj
LSF_DIR = lsf

SRC_C = $(wildcard $(SRC_DIR)/*.cpp)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJ_C = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_C))
OBJ_CU = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_CU))

CUDA_LIB_DIR = /opt/cuda/lib
CUDA_LIB = -lcudart -lcuda

LSF = $(wildcard $(LSF_DIR)/*.lsf)



all: $(TARGET)

serial: CCFLAGS += -DSERIAL
serial: NVFLAGS += -DSERIAL
serial: $(OBJ_C)
	$(CC) $(CCFLAGS) -o $@ $^	

parallel: $(OBJ_C) $(OBJ_CU)
	$(NV) $(NVFLAGS) -o $@ $^ -L$(CUDA_LIB_DIR) $(CUDA_LIB)

DEP := $(patsubst $(OBJ_DIR)/%.o, $(OBJ_DIR)/%.d, $(OBJ_C)) \
			 $(patsubst $(OBJ_DIR)/%.o, $(OBJ_DIR)/%.d, $(OBJ_CU))
-include $(DEP)
DEPFLAGS = -MMD -MF $(@:.o=.d)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	$(CC) $(CCFLAGS) -c $< -o $@ $(DEPFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(OBJ_DIR)
	$(NV) $(NVFLAGS) -c $< -o $@ $(DEPFLAGS)

clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(DEP)

fullclean:
	rm -rf $(OBJ_DIR) $(LSF_DIR) logs err serial parallel

bsubload: $(TARGET)
	for lsf_script in $(LSF) ; do \
		if command -v bsub >/dev/null 2>&1 ; then \
			bsub < $$lsf_script ; \
		else \
			sh $$lsf_script ; \
		fi \
	done

lsf:
	echo $(KBLOCKS) $(KTHREADS) $(EXEC) $(ARRAY_SIZE) $(CYCLES)
	mkdir -p lsf

	$(eval JOB_NAME := "lab4_3_$(EXEC)_$(KBLOCKS)_$(KTHREADS)")
	$(eval PROJECT_NAME := "mothm_lab4_3")
	$(eval LOG_FILE := "$(JOB_NAME).log")

	echo -e "#!/bin/bash\nmkdir -p logs err\n\n#BSUB -J $(JOB_NAME)\n#BSUB -P $(PROJECT_NAME)\n#BSUB -W 08:00\n#BSUB -n 1\n#BSUB -oo logs/$(LOG_FILE)\n#BSUB -eo err/$(LOG_FILE)\n\nexport ARRAY_SIZE=$(ARRAY_SIZE)\nexport CYCLES=$(CYCLES)\nexport KBLOCKS=$(KBLOCKS)\nexport KTHREADS=$(KTHREADS)\n\nmodule load cuda/11.4\n{ time ./$(EXEC) > logs/$(JOB_NAME).tlog ; } 2>> logs/$(JOB_NAME).tlog" > "./lsf/pr$(KBLOCKS)_$(KTHREADS)_$(EXEC).lsf"

	chmod +x ./lsf/pr$(KBLOCKS)_$(KTHREADS)_$(EXEC).lsf

auto:
	make clean
	make serial
	make clean
	make parallel

	make lsf EXEC=serial ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=1 KTHREADS=1
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=16
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=32
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=64
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=128
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=256
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=512
	make lsf EXEC=parallel ARRAY_SIZE=40000000 CYCLES=250 KBLOCKS=0 KTHREADS=1024

	make bsubload

.PHONY: lsf

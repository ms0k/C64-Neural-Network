CLANG_MOS=./llvm-mos/
ONNX2C=./onnx2c/build/onnx2c -l4

mnist.prg: mnist_q.c main_mnist_c64.c misc.h mnist/mnist.simplified.onnx
	$(CLANG_MOS)/bin/mos-c64-clang -o mnist.prg -Ofast main_mnist_c64.c mnist_q.c

mnist_test_pc: mnist_q.c main_mnist_x86.c misc.h
	gcc -o test mnist_q.c main_mnist_x86.c

disas:
	$(CLANG_MOS)/bin/llvm-objdump -d mnist.prg.elf > mnist.s

mnist_q.c: mnist/mnist.simplified.onnx
	$(ONNX2C) --quantize mnist/mnist.simplified.onnx > mnist_q.c
	sed -i 's/float alpha = 1.0000000000000000000;/int32_t alpha=1;/g' mnist_q.c
	sed -i 's/float beta = 1.0000000000000000000;/int32_t beta=1;/g' mnist_q.c
	sed -i 's/#include <math.h>//g' mnist_q.c

mnist/mnist.simplified.onnx: mnist/train.py
	cd mnist && make -f Makefile
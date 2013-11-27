all: dart
CC = g++
CFLAGS = -g $(O_LIBS)


final:
	make dart OPT=-DFINAL
step:
	make dart OPT=-DSTEP
gray_scale:
	make dart OPT=-DGRAY_SCALE
dart: face.cpp
	$(CC) $(CFLAGS) $(OPT) face.cpp
run: dart
	./run.sh $(THRESHOLD)

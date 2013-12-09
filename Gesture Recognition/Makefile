CC = g++ $(O_LIBS)
EXE = g_detect
all: main

APPLE = -DAPPLE_WEBCAM
APPLE_CAM_WAIT = 1000
apple: CONFIG_FLAGS=$(APPLE) -DAPPLE_CAM_WAIT=$(APPLE_CAM_WAIT)
main: video.cpp
	$(CC) -Wall $(CONFIG_FLAGS) video.cpp -o $(EXE)

apple: main

debug: video.cpp
	$(CC) -g video.cpp -o $(EXE)

clean: 
	rm -rf *.o $(EXE) *.dSYM

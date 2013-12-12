CC = g++ $(O_LIBS)
EXE = g_detect
all: main

APPLE = -DAPPLE_WEBCAM
APPLE_CAM_WAIT = 1000
apple: CONFIG_FLAGS=$(APPLE) -DAPPLE_CAM_WAIT=$(APPLE_CAM_WAIT)
apple_d: CONFIG_FLAGS=$(APPLE) -g -DAPPLE_CAM_WAIT=$(APPLE_CAM_WAIT) 
main: video.cpp
	$(CC) -Wall $(CONFIG_FLAGS) video.cpp -o $(EXE)

apple: main
apple_d: main

debug: video.cpp
	$(CC) -g video.cpp -o $(EXE)

clean: 
	rm -rf *.o $(EXE) *.dSYM

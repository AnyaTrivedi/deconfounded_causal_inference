EXECUTABLES = preprocessing

all: $(EXECUTABLES)

preprocessing: preprocessing.cpp
	g++ -std=c++17 -march=native preprocessing.cpp -o preprocessing

clean:
	rm -rf $(EXECUTABLES)
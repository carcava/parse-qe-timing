default : all

all :
	g++ -o parsepwout.x parsepwout.cpp
	g++ -o parsecpout.x parsecpout.cpp

clean :
	rm -f *.o *.x

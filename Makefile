

CPPINCLUDE_DIRS =  -I. -I/usr/local/opencv/include 

CPPLIBS = -L/usr/local/opencv/lib -lopencv_core -lopencv_flann -lopencv_video
LIBS =

CPP = g++
CFLAGS = -Wall -c -I.
CPPFLAGS = -Wall $(INCLUDE_DIRS)
LFLAGS = -Wall 

CPPSOURCE = auto-pilot.cpp lane_ops.cpp signal_ops.cpp traffic_ops.cpp debug.h 
CPPOUTFILE = auto-pilot


CPPOBJS = $(CPPSOURCE:.cpp=.o)

all: $(CPPOUTFILE)

$(CPPOUTFILE): $(CPPOBJS)
	$(CPP) $(CPPFLAGS) $(CPPOBJS) -o $(CPPOUTFILE) `pkg-config --libs opencv` $(CPPLIBS)

clean:
	rm -f *.o $(CPPOUTFILE)

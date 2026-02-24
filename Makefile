TARGET := bienvenue
MODULE := fonction-bienvenue
CXX = g++ -c
LD = g++ -o
RM = rm -f
CXXFLAGS = -Wall -std=c++11
LDFLAGS =
$(info Fabrication du programme : $(TARGET))
all : $(TARGET)
$(TARGET): $(TARGET).o $(MODULE).o
$(LD) $@ $(LDFLAGS) $^
$(TARGET).o: $(TARGET).cpp $(MODULE).h
$(CXX) $(CXXFLAGS) $<
$(MODULE).o: $(MODULE).cpp $(MODULE).h
$(CXX) $(CXXFLAGS) $<
.PHONY: clean
clean:
$(RM) *.o
cleanall:
$(RM) *.o $(TARGET)
rebuild: clean all

CFLAGS = -Wall -O3 `pkg-config --cflags opencv`

LIBS = `pkg-config --libs opencv`

main:test.o sgbm.o  x_cost.o cost_cal.o l_cost.o cost_init.o cost_final.o
	g++ -std=c++11 -g -o main test.o  sgbm.o x_cost.o cost_cal.o l_cost.o cost_init.o cost_final.o $(LIBS) -lpthread
cost_final.o:cost_final.s
	as -mcpu=cortex-a15 -mfpu=neon  -o  cost_final.o cost_final.s
l_cost.o:l_cost.s
	as -mcpu=cortex-a15 -mfpu=neon  -o  l_cost.o l_cost.s		
cost_init.o:cost_init.s
	as -mcpu=cortex-a15 -mfpu=neon  -o  cost_init.o cost_init.s	
x_cost.o:x_cost.s
	as -mcpu=cortex-a15 -mfpu=neon  -o  x_cost.o x_cost.s	

cost_cal.o:cost_cal.s
	as -mcpu=cortex-a15 -mfpu=neon  -o  cost_cal.o cost_cal.s

sgbm.o:sgbm.cpp
	g++ -std=c++11 -mcpu=cortex-a15 -mfpu=neon -mapcs  -marm -c $(CFLAGS) sgbm.cpp
	
test.o:test.cpp   
	g++ -std=c++11 -mcpu=cortex-a15 -mfpu=neon -mapcs  -marm -c $(CFLAGS) test.cpp 
	

.phony:clean

clean:
	rm -rf ./*.o ./a.out

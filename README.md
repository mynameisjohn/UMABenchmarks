This is a bunch of cuda code and makefiles designed to test the performance of UMA.

I basically added a compile time define to switch on the use of UMA code

=============== Update ================

I'm revisiting this after not looking at it for a while. It's a bit of a mess.

Here's how it works; cuProf.py generates data for a graph that indicates how much
time a program spends doing certain things, i.e Host to Device transfer, memory allocation,
etc. (basically the output of nvprof.) 

For the actual benchmarks, it seems I was writing things by hand into nrmData.txt, which is
unfortunate. The rodinia benchmarks used in that file are the ones that I wrote UMA code for, 
apparently, and I was not at all consistent in how I chose to split the standard / UMA code. 

It seems, generally, that I wrote two files, one being the UMA version. There are a few cases
where I used a conditional define (-DUMA) to fence the code, but not all of the rodinia Makefiles
are looking for it, so I have to assume that didn't pan out. 

It would be nice to automate this in such a way that I could run those benchmarks twice and get
the right data; I would have to either compile and execute the code twice, or make the decision
to use UMA at runtime and test the code that way. The latter makes more sense...

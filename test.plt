set key right
set offset -0.5,-0.5,0,0
set grid y
set style data histograms
set style histogram rowstacked
set boxwidth 0.5
set style fill solid 1.0 border -1
set ytics 10 nomirror
set yrange [:125]
set ylabel "Percentage of Execution Time"
set xlabel "Application"
set title "Distribution of Runtime"
set ytics 10
 
plot 'prof.dat' using \
	2 t "Kernel", \
	'' using 3 t "Mem", \
	'' using 4:xtic(1) t "HtoD", \
	'' using 5:xtic(1) t "DtoH", \
	'' using 6:xtic(1) t "API"

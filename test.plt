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
	2 t "Kernel" lc rgb '#0066CC#0066FF', \
	'' using 3 t "Mem" lc rgb '#0066FF', \
	'' using 4:xtic(1) t "HtoD" lc rgb '#FF3300', \
	'' using 5:xtic(1) t "DtoH" lc rgb '#33CC33', \
	'' using 6:xtic(1) t "API" lc rgb '#009933'

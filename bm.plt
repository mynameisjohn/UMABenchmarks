set xrange [0:100];
set xtics 100
set xtics add ("pathfinder \n19.622" 19.622)
set xtics add ("SGAC_R \n28.66" 28.66)
set xtics add ("Needle \n35.879" 35.879)
set xtics add ("gsRelax \n46.068" 46.068)
set xtics add ("srad_v2 \n62.658" 62.658)
set xtics add ("gaussian \n78.28" 78.28)
set xtics add ("lud_cuda \n 96.094" 96.094)
set yrange [0:2.5]
set ytics 0.1
set style line 1 lw 6;
set xlabel 'Percentage of Time Spent in Kernel' offset 0,-1;
set ylabel 'UMA Runtime / Traditional Runtime';
set title 'Normalized Runtimes for UMA Benchmarks';
plot 'nrmData.txt' using 1:($3/$2) with linespoints lw 4 lc rgb '#FF9933' title 'N=128';
replot 'nrmData.txt' using 1:($5/$4) with linespoints lw 4 lc rgb '#FF5050' title 'N=256';
replot 'nrmData.txt' using 1:($7/$6) with linespoints lw 4 lc rgb '#6600FF' title 'N=512';
replot 'nrmData.txt' using 1:($9/$8) with linespoints lw 4 lc rgb '#0066FF' title 'N=1024';
replot 'nrmData.txt' using 1:($11/$10) with linespoints lw 4 lc rgb '#00CC66' title 'N=2048
replot 1 title 'Normalization Line' with lines lt 2 lc rgb 'black';

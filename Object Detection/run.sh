j=0
t=30
echo "threshold: $t"
for i in `ls dartboards/dart*.jpg`; do
	new_name=$(basename $i)
	./a.out $i $t ; 
	mv output.jpg dart_out/${new_name};
	mv segmented.jpg dart_segment/${new_name};
done

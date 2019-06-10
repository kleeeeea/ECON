echo `pwd`
mkdir -p tmp
rm -rf tmp/*

RAW=${RAW:- "/disk/home/klee/data_hiercon/cs_merged_tokenized"}
CMD="python /disk/home/klee/workspace_remote/Hiercon/candidate_generation/to_json/dbpedia_extract.py 0 "
export THREAD=10

num_lines=$((`wc -l < $RAW` / $THREAD))

csplit -s -f "tmp/split_files." -n 5  $RAW $num_lines {$(($THREAD-2))}
for f in tmp/split_files.*
do
    eval "$CMD $f"
done
wait

echo 'done'
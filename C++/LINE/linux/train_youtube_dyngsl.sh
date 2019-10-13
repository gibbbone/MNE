#!/bin/sh
# common arguments for gsl library locations:
#INCLUDES="-I/$HOME/gsl/include"
#LIBS="-L/$HOME/gsl/lib"
while getopts ':i:l:' flag; do
	case "${flag}" in
	    i) 
        INCLUDES="-I${OPTARG}"
        ;;
        l) 
        libs_arg="${OPTARG}"
        LIBS="-L$libs_arg"
        ;;
        *) echo "Option not recognized"; exit 1;;
    esac
done

export LD_LIBRARY_PATH="$libs_arg":$LD_LIBRARY_PATH

g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas "$INCLUDES" "$LIBS"
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct "$INCLUDES" "$LIBS"
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result normalize.cpp -o normalize "$INCLUDES" "$LIBS"
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result concatenate.cpp -o concatenate "$INCLUDES" "$LIBS"

if [[ ! -e youtube-links.txt ]]; then
    wget http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz
    gunzip youtube-links.txt.gz
fi 

python3 preprocess_youtube.py youtube-links.txt net_youtube.txt
./reconstruct -train net_youtube.txt -output net_youtube_dense.txt -depth 2 -threshold 1000
./line -train net_youtube_dense.txt -output vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10000 -threads 40
./line -train net_youtube_dense.txt -output vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 40
./normalize -input vec_1st_wo_norm.txt -output vec_1st.txt -binary 1
./normalize -input vec_2nd_wo_norm.txt -output vec_2nd.txt -binary 1
./concatenate -input1 vec_1st.txt -input2 vec_2nd.txt -output vec_all.txt -binary 1

cd evaluate
./run.sh ../vec_all.txt
python3 score.py result.txt
cd ..

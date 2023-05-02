# TODO: split blockCV and bolivar

# BLOCKS=("ABEJORRAL" "ALEJANDRÍA" "CHIGORODÓ" "COCORNÁ" "EL CARMEN DE VIBORAL" \
#        "GRANADA" "LA UNIÓN" "NARIÑO" "SABANALARGA" "SAN CARLOS" "SAN FRANCISCO" \
#        "SAN LUIS" "SAN RAFAEL" "SAN ROQUE" "SONSÓN")
# for mpio in "${BLOCKS[@]}"
# do
#        for sub in "full"
#        do
#               python -m ood_bench.scripts.main\
#                      --n_trials 1\
#                      --dataset RELand\
#                      --envs_p 0\
#                      --envs_q 1\
#                      --backbone mlp-reland\
#                      --output_dir /home/siqiz/Landmine/ood_res/blockCV_${sub}\
#                      --data_dir . \
#                      --n_workers 1 \
#                      --split blockCV \
#                      --valid "${mpio}" \
#                      --subset ${sub}
#        done
# done

BOLIVAR=("SAN JUAN NEPOMUCENO" "CARTAGENA DE INDIAS" "CÓRDOBA" "SANTA ROSA DEL SUR" "ZAMBRANO")
for mpio in "${BOLIVAR[@]}"
do
       for sub in "full"
       do
              python -m ood_bench.scripts.main\
                     --n_trials 1\
                     --dataset RELand\
                     --envs_p 0\
                     --envs_q 1\
                     --backbone mlp-reland\
                     --output_dir /home/siqiz/Landmine/ood_res/bolivar_${sub}\
                     --data_dir . \
                     --n_workers 1 \
                     --split bolivar \
                     --valid "${mpio}" \
                     --subset ${sub} 
       done  
done
ds_range="001 003 009 027 081 243"
replid_range="a b c"

add_param() {
    tag=$1
    for exptype in pip-ind pip-seq; do
        for ds in $ds_range; do
            for replid in $replid_range; do
                path_asr=$(ls -d runs/asr$tag-ds$ds-$replid-* | xargs basename)
                path_conf=$(ls -d runs/$exptype$tag-ds$ds-$replid-*)/config.yml
                sed -i "/asr_model_dir/d" $path_conf
                echo -e "asr_model_dir\t../$path_asr" >> $path_conf
            done
        done
    done
    for exptype in pip-ind; do
        for ds in $ds_range; do
            for replid in $replid_range; do
                path_ti=$(ls -d runs/text-image$tag-ds$ds-$replid-* | xargs basename)
                path_conf=$(ls -d runs/$exptype$tag-ds$ds-$replid-*)/config.yml
                sed -i "/text_image_model_dir/d" $path_conf
                echo -e "text_image_model_dir\t../$path_ti" >> $path_conf
            done
        done
    done
}

add_param
add_param -jp

export CUDA_VISIBLE_DEVICES=0

moppit-mog-dfm \
--hf-root moPPIt \
--output_file './samples.csv' \
--length 10 \
--n_batches 600 \
--weights 1 1 1 4 4 2 \
--motifs '16-31,62-79' \
--motif_penalty \
--objectives Hemolysis Non-Fouling Half-Life Affinity Motif Specificity \
--target_protein MHVPSGAQLGLRPDLLARRRLKRCPSRWLCLSAAWSFVQVFSEPDGFTVIFSGLGNNAGGTMHWNDTRPAHFRILKVVLREAVAECLMDSYSLDVHGGRRTAAG
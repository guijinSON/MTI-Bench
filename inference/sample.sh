#!/bin/bash

python MTI_Inference_S3T.py --model_name "TheBloke/tulu-7B-fp16" --batch_size 1

python STI_Inference_S3T.py --model_name "TheBloke/tulu-7B-fp16" --batch_size 1

python Batching_Inference.py --model_name "TheBloke/tulu-7B-fp16" --batch_size 1


python MTI_Inference_S3T.py --model_name "TheBloke/tulu-13B-fp16" --batch_size 1

python STI_Inference_S3T.py --model_name "TheBloke/tulu-13B-fp16" --batch_size 1

python Batching_Inference.py --model_name "TheBloke/tulu-13B-fp16" --batch_size 1


python MTI_Inference_S3T.py --model_name "lmsys/vicuna-7b-v1.5" --batch_size 1

python STI_Inference_S3T.py --model_name "lmsys/vicuna-7b-v1.5" --batch_size 1

python Batching_Inference.py --model_name "lmsys/vicuna-7b-v1.5" --batch_size 1


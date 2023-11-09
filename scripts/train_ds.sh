debug --master_port=24999 --include localhost:3 train_ds.py --dataset_dir='../Datasets' --vision_pretrained=./model_weights/SAM/sam_vit_h_4b8939.pth --dataset="reason_seg" --sample_rates="1" --exp_name="lisa-llama2-7b" --version='/root/Workspace-new/qihuad/github/LLaVA/checkpoints/llava-7b-llama-2-7b-chat' --conv_type='llava_llama_2' --batch_size=1

debug --master_port=24999 --include localhost:3 train_ds.py --dataset_dir='../Datasets' --vision_pretrained=./model_weights/SAM/sam_vit_h_4b8939.pth --dataset="reason_seg" --sample_rates="1" --exp_name="lisa-llama2-7b" --version='/root/Workspace-new/qihuad/github/LLaVA/checkpoints/llava-llama2-7b' --conv_type='llava_llama_2' --batch_size=1

debug --master_port=24999 --include localhost:3 train_ds.py --dataset_dir='../Datasets' --vision_pretrained=./model_weights/SAM/sam_vit_h_4b8939.pth --dataset="reason_seg" --sample_rates="1" --exp_name="lisa-llama2-13b" --version='liuhaotian/llava-llama-2-13b-chat-lightning-preview' --conv_type='llava_llama_2' --batch_size=1

# lisa training
deepspeed --master_port=24999 --include localhost:0 train_ds.py \
    --dataset_dir='../lisa_dataset' \
    --vision_pretrained=./model_weights/SAM/sam_vit_h_4b8939.pth \
    --dataset="sem_seg||refer_seg||vqa||reason_seg" \
    --sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis" \
    --sample_rates="9,3,3,1" \
    --exp_name="lisa-llama2-7b" \
    --version='/root/Workspace-new/qihuad/github/LLaVA/checkpoints/llava-7b-llama-2-7b-chat' \
    --conv_type='llava_llama_2' \
    --batch_size=1

# ICS training
debug_ds --master_port=24999 \
         --include localhost:0 \
         train_ics.py \
         --dataset_dir='../lisa_dataset' \
         --vision_pretrained=./model_weights/SAM/sam_vit_h_4b8939.pth \
         --dataset="sem_seg" \
         --sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis" \
         --sample_rates="1" \
         --exp_name="ics-llama2-7b" \
         --version='/root/Workspace-new/qihuad/github/LLaVA/checkpoints/llava-7b-llama-2-7b-chat' \
         --conv_type='llava_llama_2' \
         --batch_size=2
        #  --no_eval

# debug
debug_ds --master_port=24999 \
         --include localhost:0 \
         train_ics.py \
         --dataset_dir='../lisa_dataset' \
         --vision_pretrained=./model_weights/SAM/sam_vit_h_4b8939.pth \
         --dataset="sem_seg" \
         --sem_seg_data="paco_lvis" \
         --sample_rates="1" \
         --exp_name="ics-llama2-7b" \
         --version='/root/Workspace-new/qihuad/github/LLaVA/checkpoints/llava-7b-llama-2-7b-chat' \
         --conv_type='llava_llama_2' \
         --batch_size=1 \
         -d
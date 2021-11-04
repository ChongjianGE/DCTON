#python3 /apdcephfs/share_1330111/chongjiange/code/cyclegan/test_model_general.py --dataroot /apdcephfs/share_1330111/chongjiange/data/try_on_data/256_test_data/ --model densepose_warp_hollow --name re_hole_fushion_8c_resg_wholenewskin_normDA_13_arm1_all1 --gpu_ids 0 

#python3 /apdcephfs/share_1330111/chongjiange/code/cyclegan/dense_warp_net.py --dataroot /apdcephfs/share_1330111/chongjiange/data/try_on_data/256_train_data/ --model densepose_warp_hollow --name re_hole_fushion_8c_resg_wholenewskin_normDA_13_arm10_all10  --batch_size 32 --gpu_ids 0,1,2,3,4,5,6,7

#python3 /apdcephfs/share_1330111/chongjiange/code/cyclegan/dense_to_mask_net.py --name mask_with_low_resg --model densepose_to_mask --dataroot /apdcephfs/share_1330111/chongjiange/data/try_on_data/256_train_data/ --checkpoints_dir /apdcephfs/share_1330111/chongjiange/train_output/cyclemask/checkpoints/ --batch_size 32 --gpu_ids 0,1,2,3

#python3 /apdcephfs/share_1016399/chongjiange/code/cyclegan_10/test_model_general.py --name disunified_tryon_halfa1_bzbackskinok_vggcw_1l1_9GApre/ --num_test 30000 --model disunified_tryon --dataroot /apdcephfs/share_1016399/chongjiange/data/try_on_data/256_test_data/ --checkpoints_dir /apdcephfs/share_1016399/chongjiange/train_output/cycletryon/checkpoints/ --gpu_ids 0

python3 /apdcephfs/share_1016399/chongjiange/code/cyclegan_10/test_model_general.py --name 	display_refine_1l1c_0vgg_10agback_400 --num_test 30000 --model disunified_tryon --dataroot /apdcephfs/share_1016399/chongjiange/data/try_on_data/256_test_data/ --checkpoints_dir /apdcephfs/share_1016399/chongjiange/train_output/cycletryon/checkpoints/ --gpu_ids 0



python3 disunified_net.py \
	--name display \
	--model disunified_tryon \
	--dataroot /apdcephfs/share_1016399/chongjiange/data/try_on_data/256_train_data/ \
	--checkpoints_dir /apdcephfs/share_1016399/chongjiange/train_output/cycletryon/checkpoints/ \
	--batch_size 32 \
	--gpu_ids 0,1,2,3,4,5,6,7 \
	--save_epoch_freq 10 \
	--save_latest_freq 2000


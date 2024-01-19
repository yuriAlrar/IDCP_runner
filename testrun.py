from mmagic.apis import MMagicInferencer

checkpoint = 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'
img_path = 'data/mosaic.png'
editor1 = MMagicInferencer('esrgan', model_ckpt=checkpoint)
editor1.infer(img=img_path,result_out_dir='output/esrgan.png')

checkpoint = 'https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth'
editor2 = MMagicInferencer('real_esrgan', model_ckpt=checkpoint)

editor2.infer(img=img_path,result_out_dir='output/real_esrgan.png')

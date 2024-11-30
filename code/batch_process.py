import subprocess

# 列表包含所有要處理的影片路徑
videos = [

    # "/013/edit_video/013R_sb_1_split/013R_sb_1_2.mp4",

    "/018/edit_video/018L_sa_1_split/018L_sa_1_1.mp4",
    "/018/edit_video/018L_sa_1_split/018L_sa_1_2.mp4",
    "/018/edit_video/018L_sa_1_split/018L_sa_1_3.mp4",
    "/018/edit_video/018L_sa_1_split/018L_sa_1_4.mp4",
    "/018/edit_video/018L_sa_1_split/018L_sa_1_5.mp4",

    "/018/edit_video/018L_sa_2_split/018L_sa_2_1.mp4",
    "/018/edit_video/018L_sa_2_split/018L_sa_2_2.mp4",
    "/018/edit_video/018L_sa_2_split/018L_sa_2_3.mp4",
    "/018/edit_video/018L_sa_2_split/018L_sa_2_4.mp4",
    "/018/edit_video/018L_sa_2_split/018L_sa_2_5.mp4",

    "/018/edit_video/018L_sb_1_split/018L_sb_1_1.mp4",
    "/018/edit_video/018L_sb_1_split/018L_sb_1_2.mp4",
    "/018/edit_video/018L_sb_1_split/018L_sb_1_3.mp4",
    "/018/edit_video/018L_sb_1_split/018L_sb_1_4.mp4",
    "/018/edit_video/018L_sb_1_split/018L_sb_1_5.mp4",

    "/018/edit_video/018L_sb_2_split/018L_sb_2_1.mp4",
    "/018/edit_video/018L_sb_2_split/018L_sb_2_2.mp4",
    "/018/edit_video/018L_sb_2_split/018L_sb_2_3.mp4",
    "/018/edit_video/018L_sb_2_split/018L_sb_2_4.mp4",
    "/018/edit_video/018L_sb_2_split/018L_sb_2_5.mp4",

    "/018/edit_video/018R_ha_1_split/018R_ha_1_1.mp4",
    "/018/edit_video/018R_ha_1_split/018R_ha_1_2.mp4",
    "/018/edit_video/018R_ha_1_split/018R_ha_1_3.mp4",
    "/018/edit_video/018R_ha_1_split/018R_ha_1_4.mp4",
    "/018/edit_video/018R_ha_1_split/018R_ha_1_5.mp4",

    "/018/edit_video/018R_ha_2_split/018R_ha_2_1.mp4",
    "/018/edit_video/018R_ha_2_split/018R_ha_2_2.mp4",
    "/018/edit_video/018R_ha_2_split/018R_ha_2_3.mp4",
    "/018/edit_video/018R_ha_2_split/018R_ha_2_4.mp4",
    "/018/edit_video/018R_ha_2_split/018R_ha_2_5.mp4",
    "/018/edit_video/018R_ha_2_split/018R_ha_2_6.mp4",
    "/018/edit_video/018R_ha_2_split/018R_ha_2_7.mp4",

    "/018/edit_video/018R_hb_1_split/018R_hb_1_1.mp4",
    "/018/edit_video/018R_hb_1_split/018R_hb_1_2.mp4",
    "/018/edit_video/018R_hb_1_split/018R_hb_1_3.mp4",
    "/018/edit_video/018R_hb_1_split/018R_hb_1_4.mp4",
    "/018/edit_video/018R_hb_1_split/018R_hb_1_5.mp4",

    "/018/edit_video/018R_hb_2_split/018R_hb_2_1.mp4",
    "/018/edit_video/018R_hb_2_split/018R_hb_2_2.mp4",
    "/018/edit_video/018R_hb_2_split/018R_hb_2_3.mp4",
    "/018/edit_video/018R_hb_2_split/018R_hb_2_4.mp4",
    "/018/edit_video/018R_hb_2_split/018R_hb_2_5.mp4",
    "/018/edit_video/018R_hb_2_split/018R_hb_2_6.mp4",
    "/018/edit_video/018R_hb_2_split/018R_hb_2_7.mp4",


]

# 迴圈遍歷每個影片並執行命令
for video in videos:
    print(f'current video path: {video}')
    command = ["python", "demo/vis.py", "--video", video]
    subprocess.run(command)


# python batch_process.py


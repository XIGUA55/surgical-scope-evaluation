# 胰腺手术扶镜质量评价复现报告

## 结果性质

完整单病例分段留出实验，不代表跨病例泛化结果。

## 数据审计

```json
{
  "videos": [
    {
      "path": "/root/autodl-tmp/surgical/data/video/VID001.mp4",
      "case_id": "VID001",
      "segment_id": "VID001",
      "readable": true,
      "width": 1920,
      "height": 1080,
      "fps": 25.0,
      "frame_count": 15003,
      "duration_s": 600.189388,
      "file_size": 1267879712,
      "truncated_atom": null,
      "declared_atom_end": null,
      "error": null
    },
    {
      "path": "/root/autodl-tmp/surgical/data/video/VID001_B_5fps.mp4",
      "case_id": "VID001",
      "segment_id": "VID001_B",
      "readable": true,
      "width": 1920,
      "height": 1080,
      "fps": 5.0,
      "frame_count": 3000,
      "duration_s": 600.0,
      "file_size": 808459108,
      "truncated_atom": null,
      "declared_atom_end": null,
      "error": null
    },
    {
      "path": "/root/autodl-tmp/surgical/data/video/VID001_C_5fps.mp4",
      "case_id": "VID001",
      "segment_id": "VID001_C",
      "readable": true,
      "width": 1920,
      "height": 1080,
      "fps": 5.0,
      "frame_count": 3000,
      "duration_s": 600.0,
      "file_size": 777178912,
      "truncated_atom": null,
      "declared_atom_end": null,
      "error": null
    },
    {
      "path": "/root/autodl-tmp/surgical/data/video/VID001_D_5fps.mp4",
      "case_id": "VID001",
      "segment_id": "VID001_D",
      "readable": true,
      "width": 1920,
      "height": 1080,
      "fps": 5.0,
      "frame_count": 3001,
      "duration_s": 600.2,
      "file_size": 738938590,
      "truncated_atom": null,
      "declared_atom_end": null,
      "error": null
    }
  ],
  "label_sets": [
    {
      "path": "/root/autodl-tmp/surgical/data/label/VID001_Bseg",
      "case_id": "VID001",
      "segment_id": "VID001_B",
      "label_files": 1456,
      "annotation_rows": 1456,
      "minimum_frame": 1120,
      "maximum_frame": 2907,
      "object_ids": {
        "0": 1456
      },
      "status": "ready"
    },
    {
      "path": "/root/autodl-tmp/surgical/data/label/VID001_Cseg",
      "case_id": "VID001",
      "segment_id": "VID001_C",
      "label_files": 1882,
      "annotation_rows": 1882,
      "minimum_frame": 65,
      "maximum_frame": 2920,
      "object_ids": {
        "0": 1871,
        "1": 11
      },
      "status": "ready"
    },
    {
      "path": "/root/autodl-tmp/surgical/data/label/VID001_Dseg",
      "case_id": "VID001",
      "segment_id": "VID001_D",
      "label_files": 515,
      "annotation_rows": 515,
      "minimum_frame": 40,
      "maximum_frame": 2538,
      "object_ids": {
        "0": 515
      },
      "status": "ready"
    }
  ],
  "errors": [],
  "summary": {
    "video_count": 4,
    "readable_video_count": 4,
    "label_file_count": 3853,
    "ready_label_file_count": 3853,
    "error_count": 0
  }
}
```

## RT-DETRv2 训练

```json
[
  {
    "epoch": 1,
    "train_loss": 19.21141001156398,
    "val_map": 0.09896642714738846,
    "val_map_50": 0.2976143956184387,
    "val_map_75": 0.04361346364021301,
    "val_map_small": -1.0,
    "val_map_medium": 0.002603045664727688,
    "val_map_large": 0.10190369188785553,
    "val_mar_1": 0.2187035083770752,
    "val_mar_10": 0.36370882391929626,
    "val_mar_100": 0.40196600556373596,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.07500000298023224,
    "val_mar_large": 0.4040641784667969,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.496291791304408
  },
  {
    "epoch": 2,
    "train_loss": 12.724250203960544,
    "val_map": 0.08894385397434235,
    "val_map_50": 0.23642003536224365,
    "val_map_75": 0.04793408513069153,
    "val_map_small": -1.0,
    "val_map_medium": 0.0018611873965710402,
    "val_map_large": 0.0976811945438385,
    "val_mar_1": 0.23156216740608215,
    "val_mar_10": 0.4314027726650238,
    "val_mar_100": 0.47045695781707764,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.19166666269302368,
    "val_mar_large": 0.472245991230011,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.204280631557392
  },
  {
    "epoch": 3,
    "train_loss": 12.002378272486258,
    "val_map": 0.19910380244255066,
    "val_map_50": 0.5231977701187134,
    "val_map_75": 0.10814015567302704,
    "val_map_small": -1.0,
    "val_map_medium": 0.00525892386212945,
    "val_map_large": 0.20528802275657654,
    "val_mar_1": 0.25913920998573303,
    "val_mar_10": 0.4156748056411743,
    "val_mar_100": 0.47088202834129333,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.18333333730697632,
    "val_mar_large": 0.4727272689342499,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.465428760350383
  },
  {
    "epoch": 4,
    "train_loss": 11.534416143710796,
    "val_map": 0.15565600991249084,
    "val_map_50": 0.39421263337135315,
    "val_map_75": 0.09171497076749802,
    "val_map_small": -1.0,
    "val_map_medium": 0.006708936300128698,
    "val_map_large": 0.16148120164871216,
    "val_mar_1": 0.2802869379520416,
    "val_mar_10": 0.4404888451099396,
    "val_mar_100": 0.4917640686035156,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.1666666716337204,
    "val_mar_large": 0.4938502609729767,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.466371501833517
  },
  {
    "epoch": 5,
    "train_loss": 11.408529158476945,
    "val_map": 0.14388184249401093,
    "val_map_50": 0.36299270391464233,
    "val_map_75": 0.08573838323354721,
    "val_map_small": -1.0,
    "val_map_medium": 0.008294410072267056,
    "val_map_large": 0.1539885252714157,
    "val_mar_1": 0.2537725865840912,
    "val_mar_10": 0.4180658757686615,
    "val_mar_100": 0.47529223561286926,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.18333333730697632,
    "val_mar_large": 0.4771657884120941,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.813312872706451
  },
  {
    "epoch": 6,
    "train_loss": 10.856109087283794,
    "val_map": 0.2243252843618393,
    "val_map_50": 0.5444511771202087,
    "val_map_75": 0.1418089121580124,
    "val_map_small": -1.0,
    "val_map_medium": 0.04022175073623657,
    "val_map_large": 0.22848442196846008,
    "val_mar_1": 0.27481403946876526,
    "val_mar_10": 0.40281614661216736,
    "val_mar_100": 0.4602019190788269,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.10833333432674408,
    "val_mar_large": 0.46245989203453064,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.935462227799077
  },
  {
    "epoch": 7,
    "train_loss": 9.990356622161446,
    "val_map": 0.26354706287384033,
    "val_map_50": 0.6158666610717773,
    "val_map_75": 0.1661766767501831,
    "val_map_small": -1.0,
    "val_map_medium": 0.03435390815138817,
    "val_map_large": 0.2671297788619995,
    "val_mar_1": 0.3034006357192993,
    "val_mar_10": 0.4430924654006958,
    "val_mar_100": 0.48740702867507935,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.19166666269302368,
    "val_mar_large": 0.4893048107624054,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.143454462606183
  },
  {
    "epoch": 8,
    "train_loss": 9.652882369010003,
    "val_map": 0.26393982768058777,
    "val_map_50": 0.6128539443016052,
    "val_map_75": 0.17564956843852997,
    "val_map_small": -1.0,
    "val_map_medium": 0.053066909313201904,
    "val_map_large": 0.26768097281455994,
    "val_mar_1": 0.30515408515930176,
    "val_mar_10": 0.46487778425216675,
    "val_mar_100": 0.5209882855415344,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.24166665971279144,
    "val_mar_large": 0.5227807760238647,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.141916754392556
  },
  {
    "epoch": 9,
    "train_loss": 9.23321193260151,
    "val_map": 0.27307647466659546,
    "val_map_50": 0.6094604134559631,
    "val_map_75": 0.19674493372440338,
    "val_map_small": -1.0,
    "val_map_medium": 0.08196883648633957,
    "val_map_large": 0.2777574062347412,
    "val_mar_1": 0.31115832924842834,
    "val_mar_10": 0.4340595006942749,
    "val_mar_100": 0.4968118965625763,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.21666666865348816,
    "val_mar_large": 0.49860963225364685,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.038059389515288
  },
  {
    "epoch": 10,
    "train_loss": 9.001597932406835,
    "val_map": 0.25991570949554443,
    "val_map_50": 0.6166854500770569,
    "val_map_75": 0.16653814911842346,
    "val_map_small": -1.0,
    "val_map_medium": 0.07658577710390091,
    "val_map_large": 0.26345640420913696,
    "val_mar_1": 0.2984059453010559,
    "val_mar_10": 0.42964932322502136,
    "val_mar_100": 0.48172158002853394,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.09166666865348816,
    "val_mar_large": 0.4842245876789093,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.382773443898325
  },
  {
    "epoch": 11,
    "train_loss": 8.596779797103379,
    "val_map": 0.26106491684913635,
    "val_map_50": 0.6101143956184387,
    "val_map_75": 0.18390043079853058,
    "val_map_small": -1.0,
    "val_map_medium": 0.007343966979533434,
    "val_map_large": 0.26514679193496704,
    "val_mar_1": 0.3090860843658447,
    "val_mar_10": 0.423485666513443,
    "val_mar_100": 0.46833157539367676,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.10000000149011612,
    "val_mar_large": 0.4706951975822449,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.363082972301799
  },
  {
    "epoch": 12,
    "train_loss": 8.436827988414974,
    "val_map": 0.2637130320072174,
    "val_map_50": 0.6337624192237854,
    "val_map_75": 0.17493322491645813,
    "val_map_small": -1.0,
    "val_map_medium": 0.07613734900951385,
    "val_map_large": 0.26828882098197937,
    "val_mar_1": 0.307970255613327,
    "val_mar_10": 0.4184909760951996,
    "val_mar_100": 0.4749734401702881,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.14166666567325592,
    "val_mar_large": 0.4771122932434082,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 13.235644507560002
  },
  {
    "epoch": 13,
    "train_loss": 8.21223192686563,
    "val_map": 0.26257237792015076,
    "val_map_50": 0.572263777256012,
    "val_map_75": 0.20133161544799805,
    "val_map_small": -1.0,
    "val_map_medium": 0.012785671278834343,
    "val_map_large": 0.26725640892982483,
    "val_mar_1": 0.3001062572002411,
    "val_mar_10": 0.4063761830329895,
    "val_mar_100": 0.45111584663391113,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.125,
    "val_mar_large": 0.4532085657119751,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.106784013679325
  },
  {
    "epoch": 14,
    "train_loss": 8.120493436907674,
    "val_map": 0.2272043526172638,
    "val_map_50": 0.5709775686264038,
    "val_map_75": 0.1297725886106491,
    "val_map_small": -1.0,
    "val_map_medium": 0.032628532499074936,
    "val_map_large": 0.23172937333583832,
    "val_mar_1": 0.2743358016014099,
    "val_mar_10": 0.37178534269332886,
    "val_mar_100": 0.4281083941459656,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.10000000149011612,
    "val_mar_large": 0.43021389842033386,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.35409896034597
  },
  {
    "epoch": 15,
    "train_loss": 7.818479235355671,
    "val_map": 0.24617108702659607,
    "val_map_50": 0.5666757822036743,
    "val_map_75": 0.1643708497285843,
    "val_map_small": -1.0,
    "val_map_medium": 0.004820395726710558,
    "val_map_large": 0.25174060463905334,
    "val_mar_1": 0.28427204489707947,
    "val_mar_10": 0.3925611078739166,
    "val_mar_100": 0.457863986492157,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.15000000596046448,
    "val_mar_large": 0.4598395824432373,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.166496124490305
  },
  {
    "epoch": 16,
    "train_loss": 7.587931335627378,
    "val_map": 0.23683540523052216,
    "val_map_50": 0.5541330575942993,
    "val_map_75": 0.15647554397583008,
    "val_map_small": -1.0,
    "val_map_medium": 0.011749373748898506,
    "val_map_large": 0.24162015318870544,
    "val_mar_1": 0.2768331468105316,
    "val_mar_10": 0.3802337944507599,
    "val_mar_100": 0.4393730163574219,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.10000000149011612,
    "val_mar_large": 0.4415507912635803,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.432148965301028
  },
  {
    "epoch": 17,
    "train_loss": 7.429141978641133,
    "val_map": 0.242707297205925,
    "val_map_50": 0.5531339049339294,
    "val_map_75": 0.17310483753681183,
    "val_map_small": -1.0,
    "val_map_medium": 0.019331060349941254,
    "val_map_large": 0.24785394966602325,
    "val_mar_1": 0.283687561750412,
    "val_mar_10": 0.3862380385398865,
    "val_mar_100": 0.4529755711555481,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.2083333283662796,
    "val_mar_large": 0.4545454680919647,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.493232487113612
  },
  {
    "epoch": 18,
    "train_loss": 7.299627445556305,
    "val_map": 0.2443487048149109,
    "val_map_50": 0.5784597396850586,
    "val_map_75": 0.16094444692134857,
    "val_map_small": -1.0,
    "val_map_medium": 0.0882592424750328,
    "val_map_large": 0.24879465997219086,
    "val_mar_1": 0.2852284908294678,
    "val_mar_10": 0.3861849009990692,
    "val_mar_100": 0.43772581219673157,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.1666666716337204,
    "val_mar_large": 0.4394652545452118,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.480657759239213
  },
  {
    "epoch": 19,
    "train_loss": 7.001126725595076,
    "val_map": 0.2492707222700119,
    "val_map_50": 0.5482175946235657,
    "val_map_75": 0.18516282737255096,
    "val_map_small": -1.0,
    "val_map_medium": 0.01712571084499359,
    "val_map_large": 0.25376608967781067,
    "val_mar_1": 0.28384697437286377,
    "val_mar_10": 0.38586610555648804,
    "val_mar_100": 0.43719446659088135,
    "val_mar_small": -1.0,
    "val_mar_medium": 0.14166666567325592,
    "val_mar_large": 0.43909090757369995,
    "val_map_per_class": -1.0,
    "val_mar_100_per_class": -1.0,
    "val_classes": 0.0,
    "val_loss": 14.671100993571514
  }
]
```

## 正式 train/val/test 指标

```json
{
  "train": {
    "map": 0.8337733149528503,
    "map_50": 1.0,
    "map_75": 0.9817439913749695,
    "map_small": -1.0,
    "map_medium": 0.682145357131958,
    "map_large": 0.8353012800216675,
    "mar_1": 0.8642170429229736,
    "mar_10": 0.8714285492897034,
    "mar_100": 0.8739697933197021,
    "mar_small": -1.0,
    "mar_medium": 0.7400000095367432,
    "mar_large": 0.8748962879180908,
    "map_per_class": -1.0,
    "mar_100_per_class": -1.0,
    "classes": 0.0,
    "loss": 3.8150403709202023
  },
  "val": {
    "map": 0.27306804060935974,
    "map_50": 0.6095817685127258,
    "map_75": 0.1965893656015396,
    "map_small": -1.0,
    "map_medium": 0.08196716010570526,
    "map_large": 0.27772295475006104,
    "mar_1": 0.3113177418708801,
    "mar_10": 0.4337407052516937,
    "mar_100": 0.49686503410339355,
    "mar_small": -1.0,
    "mar_medium": 0.2083333283662796,
    "mar_large": 0.4987165629863739,
    "map_per_class": -1.0,
    "mar_100_per_class": -1.0,
    "classes": 0.0,
    "loss": 13.026734859256422
  },
  "test": {
    "map": 0.09402239322662354,
    "map_50": 0.3118685483932495,
    "map_75": 0.03335218131542206,
    "map_small": -1.0,
    "map_medium": -1.0,
    "map_large": 0.09796003997325897,
    "mar_1": 0.17786407470703125,
    "mar_10": 0.36446601152420044,
    "mar_100": 0.4438835084438324,
    "mar_small": -1.0,
    "mar_medium": -1.0,
    "mar_large": 0.4438835084438324,
    "map_per_class": -1.0,
    "mar_100_per_class": -1.0,
    "classes": 0.0,
    "loss": 15.003240790733924
  }
}
```

## 三段完整视频推理

```json
{
  "VID001_B": {
    "case_id": "VID001",
    "segment_id": "VID001_B",
    "frames": 3000,
    "fps": 5.0,
    "inference_seconds": 33.34868563711643,
    "model_fps_excluding_io": 89.95856786214863,
    "device": "cuda",
    "rendered_video": null
  },
  "VID001_C": {
    "case_id": "VID001",
    "segment_id": "VID001_C",
    "frames": 3000,
    "fps": 5.0,
    "inference_seconds": 33.478967770934105,
    "model_fps_excluding_io": 89.60849750584458,
    "device": "cuda",
    "rendered_video": null
  },
  "VID001_D": {
    "case_id": "VID001",
    "segment_id": "VID001_D",
    "frames": 3001,
    "fps": 5.0,
    "inference_seconds": 33.510443694889545,
    "model_fps_excluding_io": 89.55417085264266,
    "device": "cuda",
    "rendered_video": null
  }
}
```

## 扶镜轨迹指标

```json
[
  {
    "case_id": "VID001",
    "segment_id": "VID001_B",
    "frame_count": 3000,
    "duration_s": 599.8,
    "detection_rate": 0.7706666666666667,
    "dominant_track_coverage_rate": 0.0166666666666666,
    "trajectory_metrics_reliable": false,
    "centered_rate_detected": 0.96,
    "mean_center_distance": 0.0874931512336355,
    "p95_center_distance": 0.1660046306979408,
    "out_of_view_rate": 0.0,
    "mean_speed": 0.0997385397282168,
    "p95_speed": 0.2254373017473606,
    "mean_acceleration": 0.275582271789847,
    "mean_abs_jerk": 0.4789340384778953,
    "jitter_rms": 0.0119048634455882,
    "recenter_event_count": 0,
    "mean_recenter_latency_s": NaN,
    "detection_gap_count": 70,
    "longest_detection_gap_s": 22.799999999998704,
    "dominant_track_gap_count": 5,
    "longest_dominant_track_gap_s": 343.79999999998046
  },
  {
    "case_id": "VID001",
    "segment_id": "VID001_C",
    "frame_count": 3000,
    "duration_s": 599.8,
    "detection_rate": 0.6296666666666667,
    "dominant_track_coverage_rate": 0.0296666666666666,
    "trajectory_metrics_reliable": false,
    "centered_rate_detected": 0.5056179775280899,
    "mean_center_distance": 0.2051634273780923,
    "p95_center_distance": 0.3295399753127046,
    "out_of_view_rate": 0.0,
    "mean_speed": 0.0903733217057962,
    "p95_speed": 0.2050255196989844,
    "mean_acceleration": 0.281566317000667,
    "mean_abs_jerk": 0.4675705336059941,
    "jitter_rms": 0.0094219476957901,
    "recenter_event_count": 3,
    "mean_recenter_latency_s": 2.733333333333339,
    "detection_gap_count": 312,
    "longest_detection_gap_s": 17.599999999999,
    "dominant_track_gap_count": 5,
    "longest_dominant_track_gap_s": 395.1999999999776
  },
  {
    "case_id": "VID001",
    "segment_id": "VID001_D",
    "frame_count": 3001,
    "duration_s": 600.0,
    "detection_rate": 0.3638787070976341,
    "dominant_track_coverage_rate": 0.0106631122959013,
    "trajectory_metrics_reliable": false,
    "centered_rate_detected": 1.0,
    "mean_center_distance": 0.0574688592868932,
    "p95_center_distance": 0.1003385743261216,
    "out_of_view_rate": 0.0,
    "mean_speed": 0.052726101478286,
    "p95_speed": 0.1091920089486367,
    "mean_acceleration": 0.1463481325276587,
    "mean_abs_jerk": 0.1945128616867806,
    "jitter_rms": 0.0063630536454602,
    "recenter_event_count": 0,
    "mean_recenter_latency_s": NaN,
    "detection_gap_count": 369,
    "longest_detection_gap_s": 26.599999999998488,
    "dominant_track_gap_count": 7,
    "longest_dominant_track_gap_s": 583.7999999999668
  }
]
```

## 结果有效性判定

```json
{
  "model_quality_gate_passed": false,
  "gate": "held-out test mAP50 >= 0.50 and mAP50-95 >= 0.25 (engineering minimum, not a clinical threshold)",
  "interpretation": "模型未达到最低工程门槛；轨迹指标仅验证计算流程，不能用于比较真实扶镜质量。",
  "independent_case_generalization": false,
  "reason": "B/C/D 均来自同一病例 VID001。"
}
```

## 检测误差分析

```json
{
  "checkpoint": "artifacts/runs/formal/best",
  "threshold_selected_on_validation": 0.5299999999999999,
  "validation_top1_iou50": {
    "threshold": 0.5299999999999999,
    "tp": 1033,
    "fp": 499,
    "fn": 849,
    "precision": 0.6742819843342036,
    "recall": 0.5488841657810839,
    "f1": 0.6051552431165788
  },
  "validation_top1_iou75": {
    "threshold": 0.5299999999999999,
    "tp": 515,
    "fp": 1017,
    "fn": 1367,
    "precision": 0.33616187989556134,
    "recall": 0.2736450584484591,
    "f1": 0.30169888693614527
  },
  "test_top1_iou50": {
    "threshold": 0.5299999999999999,
    "tp": 170,
    "fp": 156,
    "fn": 345,
    "precision": 0.5214723926380368,
    "recall": 0.3300970873786408,
    "f1": 0.4042806183115339
  },
  "test_top1_iou75": {
    "threshold": 0.5299999999999999,
    "tp": 50,
    "fp": 276,
    "fn": 465,
    "precision": 0.15337423312883436,
    "recall": 0.0970873786407767,
    "f1": 0.11890606420927466
  },
  "validation_mean_top_iou": 0.5146214649688111,
  "test_mean_top_iou": 0.4107701645318636,
  "test_error_counts": {
    "missed_low_confidence": 189,
    "localization_iou_below_050": 156,
    "localization_iou_050_to_075": 120,
    "correct_iou_075": 50
  },
  "test_group_diagnostics": [
    {
      "dimension": "area_bucket",
      "bucket": "small",
      "n": 285,
      "mean_top_iou": 0.4212277922191118,
      "threshold": 0.5299999999999999,
      "tp": 117,
      "fp": 81,
      "fn": 168,
      "precision": 0.5909090909090909,
      "recall": 0.4105263157894737,
      "f1": 0.484472049689441
    },
    {
      "dimension": "area_bucket",
      "bucket": "medium",
      "n": 103,
      "mean_top_iou": 0.46023720289462977,
      "threshold": 0.5299999999999999,
      "tp": 36,
      "fp": 17,
      "fn": 67,
      "precision": 0.6792452830188679,
      "recall": 0.34951456310679613,
      "f1": 0.46153846153846156
    },
    {
      "dimension": "area_bucket",
      "bucket": "large",
      "n": 127,
      "mean_top_iou": 0.3471833232544568,
      "threshold": 0.5299999999999999,
      "tp": 17,
      "fp": 58,
      "fn": 110,
      "precision": 0.22666666666666666,
      "recall": 0.13385826771653545,
      "f1": 0.16831683168316833
    },
    {
      "dimension": "position_bucket",
      "bucket": "central",
      "n": 498,
      "mean_top_iou": 0.41380437708929196,
      "threshold": 0.5299999999999999,
      "tp": 168,
      "fp": 155,
      "fn": 330,
      "precision": 0.5201238390092879,
      "recall": 0.3373493975903614,
      "f1": 0.4092570036540804
    },
    {
      "dimension": "position_bucket",
      "bucket": "edge",
      "n": 17,
      "mean_top_iou": 0.3218855849083732,
      "threshold": 0.5299999999999999,
      "tp": 2,
      "fp": 1,
      "fn": 15,
      "precision": 0.6666666666666666,
      "recall": 0.11764705882352941,
      "f1": 0.2
    },
    {
      "dimension": "time_quartile",
      "bucket": "Q1",
      "n": 129,
      "mean_top_iou": 0.3502726167613684,
      "threshold": 0.5299999999999999,
      "tp": 38,
      "fp": 54,
      "fn": 91,
      "precision": 0.41304347826086957,
      "recall": 0.29457364341085274,
      "f1": 0.3438914027149321
    },
    {
      "dimension": "time_quartile",
      "bucket": "Q2",
      "n": 129,
      "mean_top_iou": 0.40581132356048555,
      "threshold": 0.5299999999999999,
      "tp": 34,
      "fp": 32,
      "fn": 95,
      "precision": 0.5151515151515151,
      "recall": 0.26356589147286824,
      "f1": 0.3487179487179487
    },
    {
      "dimension": "time_quartile",
      "bucket": "Q3",
      "n": 128,
      "mean_top_iou": 0.440299820207656,
      "threshold": 0.5299999999999999,
      "tp": 40,
      "fp": 36,
      "fn": 88,
      "precision": 0.5263157894736842,
      "recall": 0.3125,
      "f1": 0.39215686274509803
    },
    {
      "dimension": "time_quartile",
      "bucket": "Q4",
      "n": 129,
      "mean_top_iou": 0.44692580965744666,
      "threshold": 0.5299999999999999,
      "tp": 58,
      "fp": 34,
      "fn": 71,
      "precision": 0.6304347826086957,
      "recall": 0.4496124031007752,
      "f1": 0.5248868778280543
    }
  ]
}
```

## 调试结论与下一步

1. 单类别分类头默认先验 0.5 会造成大量高置信度查询；已改为 0.01，修复后初始损失从约 204 降至约 22.5。
2. 过拟合明显：train/val/test mAP50-95 分别为 0.834 / 0.273 / 0.094。
3. D 上候选排序是主要错误之一：248 帧存在 IoU≥0.5 的候选，但最高分候选 IoU<0.5。
4. 定性抽检发现高光组织假阳性、运动模糊/画面边缘漏检，以及不同段对器械工作端和杆部的标注范围不一致。
5. IoU tracker 轨迹碎片严重；轨迹覆盖率不足 80% 时，运动学扶镜指标自动标记为不可靠。
6. 下一步优先统一标注规范并增加独立病例，其次做跨段均衡采样、尺度/模糊/高光增强、置信度-IoU排序校准和 ByteTrack/Kalman 追踪。

## 专家评分验证

```json
{
  "status": "ratings not supplied"
}
```

## 可复现性

配置：`/root/autodl-tmp/surgical/config/project.yaml`

随机种子：`42`

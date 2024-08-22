# From Regional Understanding to General Understanding: A Progressive Framework for Humanlike Comprehension of Corner Cases in Autonomous Driving Code Space

## Original Files Structure

This the original structure of our code after generation. **CODA-LM** is the original files in CODA-LM task (LINK: https://github.com/DLUT-LYZ/CODA-LM). The **test** stands for the images in the **CODA** which are required by **CODA-LM** (For details, please refer to the **CODA-LM** page). **annotations.json** is the annotations files in the **CODA** dataset (LINK: [CODA-Download (coda-dataset.github.io)](https://coda-dataset.github.io/download.html#instructions)). For running, please run **Generate.py**.

│  annotations.json
│  Generate.py
├─CODA-LM
│  ├─Mini
│  │  └─vqa_anno        
│  ├─Test
│  │  └─vqa_anno
│  ├─Train
│  │  └─vqa_anno        
│  └─Val
│      └─vqa_anno      
├─test
│  ├─images
│  │      0001.jpg
│  │      0002.jpg
│  │      ……
│  └─images_w_boxes
│          0001_object_1.jpg
│          0001_object_2.jpg
│          0001_object_3.jpg
│          ……

## Final Files Structure

This the final structure of our code after generation. The targeted files are in **Outputs**. Please make sure all steps are finished in the **Generate.py** and the validation is completed.

│  annotations.json
│  Generate.py
├─CODA-LM
│  ├─Mini
│  │  └─vqa_anno        
│  ├─Test
│  │  └─vqa_anno
│  ├─Train
│  │  └─vqa_anno        
│  └─Val
│      └─vqa_anno      
├─Outputs
│      driving_suggestion.jsonl
│      general_perception.jsonl
│      region_perception.jsonl
├─test
│  ├─images
│  │      0001.jpg
│  │      0002.jpg
│  │      ……
│  └─images_w_boxes
│          0001_object_1.jpg
│          0001_object_2.jpg
│          0001_object_3.jpg
│          ……
└─TestObjectAnalyzeImages
        

# From Regional to General: A Progressive Framework for Human-Like Comprehension of Corner Cases in Autonomous Driving Code Space

## Original Files Structure

This the original structure of our code. **CODA-LM** is the original files in CODA-LM task (LINK: https://github.com/DLUT-LYZ/CODA-LM). The **test** stands for the images in the **CODA** which are required by **CODA-LM** (For details, please refer to the **CODA-LM** page). **annotations.json** is the annotations files in the **CODA** dataset (LINK: [CODA-Download (coda-dataset.github.io)](https://coda-dataset.github.io/download.html#instructions)). Before running, please run 
```
pip install -r requirements.txt
python Generate.py --openai-key [YOUR OPENAI KEY] --model [gpt-4o] (--base-url "if any")
```
The original file structure:
```
.
|-- CODA-LM
|   |-- Mini
|   |-- Test
|   |-- Train
|   `-- Val
|-- Generate.py
|-- README.md
|-- annotations.json
|-- requirements.txt
`-- test
    |-- images
    `-- images_w_boxes
```

## Final Files Structure

This the final structure of our code after generation. The targeted files are in **Outputs**. Please make sure all steps are finished in the **Generate.py** and the validation is completed.

```
`
|-- CODA-LM
|   |-- Mini
|   |-- Test
|   |-- Train
|   `-- Val
|-- Generate.py
|-- Outputs
|-- README.md
|-- TestObjectAnalyzeImages
|-- annotations.json
|-- requirements.txt
`-- test
    |-- images
    `-- images_w_boxes
```

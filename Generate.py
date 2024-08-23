import os
import json
from openai import OpenAI
import argparse
import glob
import json
from shutil import copyfile, move
import cv2
from tqdm import tqdm
import pandas as pd

kinds ={'vehicles':'vehicles (cars, trucks, buses, etc.)', 
        'people':'vulnerable road users (pedestrians, cyclists, motorcyclists)',
        'traffic_signs':'traffic signs (no parking, warning, directional, etc.)', 
        'traffic_lights':'traffic lights (red, green, yellow)', 
        'traffic_cones':'traffic cones', 
        'barriers':'barriers', 
        'miscellaneous':'miscellaneous(debris, dustbin, animals, etc.)'}   
model_prompt = 'You are an autonomous driving expert, specializing in recognizing traffic scenes and making driving decisions. '
suggestion_prompt = 'Based on the general perception:{} and region perception: {},'
GeneralPerceptionPrompt = 'Please focus on the [The start of the objects]{} [The end of the objects]in the picture, and the image, please gather all the useful information and the image above and provide a detailed description of the gerneral perception and explain why the objects in the image will affect the driving. The answer should contain all the infomation and you should not contain any words directly from the prompt. If there are trafficlights in the image, please add the detailed information about the trafficlights and their colors and their influence to driving. If there are anything else that may affect the driving in the image, please add the information about the other objects and their influence to driving.\n'
ScoreImportancePrompt = '''
You are an impartial judge tasked with analyzing whether the object in the traffic scene may affect safe driving of the ego car.
You will analyze the object in the rectangle of the image, focusing on the description of the object provided by autonomous driving AI assistant[The Start of the Description]\n{}\n[The End of the Description].
 Do not allow the length of the predicted text to influence your evaluation. Maximize your text comprehension capabilities to freely match objects with high similarity, 
appropriately ignoring the relative positions and color attributes of the objects. 
You must rate the influence on a scale from 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[10]]\".
Here is the rating rules.
Rating 10 means that the object will definitely affect the ego car's safe driving, the situation is urgent to be taken into account and there is a large probability that the accident will happened.
Rating 5 means that the object will probably affect the ego car's safe driving but the accident can be avoided.
Rating 1 means that the object will not affect the ego car's driving.
'''
ObjectAnalyzeQuestion = "Please recognize and describe the object inside the red rectangle in the image in detail and explain why it affect ego car driving."
SpecificCarQuestion = 'Please analyze the type of the car in the image carefully and figure out whether there are anything special happened(like blaking light, etc.) in the car.'

class CODALMTask:
    def __init__(self, args):
        self.OPENAI_API_KEY = args.openai_key
        self.model = args.model
        self.base_url = args.base_url

    def SortJsonlFile(self, file, key):
        data = []
        assert 'jsonl' in file
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        data.sort(key=key)
        with open(file, 'w') as f:
            for line in data:
                f.write(json.dumps(line) + '\n')

    def openai_inference(self, image, question, temperature=0.2, max_tokens=2000):
        import base64
        def encode_image(image_file):
            with open(image_file, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        base64_image = encode_image(image)
        OPENAI_API_KEY = self.OPENAI_API_KEY
        MODEL = self.model
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=self.base_url)
        response = client.chat.completions.create(
            model = MODEL,
            messages = [
                {"role": "system", "content": 'You are a helpful assistant in car drving.'},
                {"role": "user", "content": [
                    {'type': 'text', 'text':question},
                    {'type': 'image_url', 'image_url': {
                        'url': f'data:image/jpg;base64,{base64_image}'
                    }}
                ]
                },
            ],
            temperature = temperature,
            max_tokens = max_tokens
        )
        result = response.choices[0].message.content
        return result   
        # return '[[9]]'

    def IgnoreGenerated(self, files, jsonl):
            data = []
            temp = files.copy()
            with open(jsonl, 'r') as f:
                data = [json.loads(line) for line in f]
            for d in data:
                name = d['image'].replace('test/images/', '')
                
                for file in files:
                    if name in file:
                        temp.remove(file)
            return temp
    
    def general_inference(self):
        def AnalyzeEveryObjectInCODA():
            def FilterImage():
                '''
                This function is used to filter out the images that are used for Test set.
                '''
                def CreateNewImages(image, save_dir, categories):
                    '''
                    This function is used to create new images According to the annotations in CODA dataset.
                    '''
                    def toImage(id):
                        return '0'*(4-len(str(id))) + str(id) + '.jpg'
                    def name(id,idx):
                        return '0'*(4-len(str(id))) + str(id) + '_{}.jpg'.format(idx)
                    with open('annotations.json', 'r') as f:
                        annotations = json.load(f)['annotations']
                        idx = 1
                        for data in annotations:
                            if '0'*(4-len(str(data['image_id']))) + str(data['image_id']) in image:
                                img = cv2.imread(os.path.join('test/images',toImage(data['image_id'])))
                                img = cv2.rectangle(img, (data['bbox'][0], data['bbox'][1]), (data['bbox'][0]+data['bbox'][2], data['bbox'][1]+data['bbox'][3]), (0, 0, 255), 2)
                                cv2.imwrite(os.path.join(save_dir, name(data['image_id'], idx)), img)
                                with open(os.path.join(save_dir, name(data['image_id'], idx)).replace('.jpg', '.txt'), 'w') as f:
                                    f.write(categories[data['category_id']])
                                idx += 1
                with open('annotations.json', 'r') as f:
                    annotations = json.load(f)
                    categories = {}
                    for data in annotations['categories']:
                        categories[data['id']] = data['name']
                PATH = 'TestObjectAnalyzeImages'
                with open('CODA-LM/Test/vqa_anno/general_perception.jsonl', 'r') as f:
                    data = [json.loads(line) for line in f]
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                for d in tqdm(data):
                    image = d['image']
                    CreateNewImages(image, PATH, categories)
            def AnalyzeTask(image, question, temperature=0.2):
                '''
                This function is used to analyze the object in the image. For multiprocessing.
                '''
                result = self.openai_inference(image, question, temperature)
                with open(image.replace('.jpg', '.json'),'w') as f:
                    json.dump({'answer':result}, f)
            
            if not os.path.exists('TestObjectAnalyzeImages') or len(glob.glob('TestObjectAnalyzeImages/*.jpg')) < 4178:
                FilterImage()
            files = glob.glob('TestObjectAnalyzeImages/*.jpg')
            for file in tqdm(files):
                with open(file.replace('.jpg', '.txt'), 'r') as f:
                    name = f.readline()
                    if os.path.exists(file.replace('.jpg', '.json')):
                        continue
                    if 'car' in name.lower():
                        question = ObjectAnalyzeQuestion + ' The object in the rectangle is {}.'.format(name) + SpecificCarQuestion
                        AnalyzeTask(file, question, 0.2)
                    else:
                        question = ObjectAnalyzeQuestion + ' The object in the rectangle is {}.'.format(name)
                        AnalyzeTask(file, question, 0.2)

        def SelfAssessment(files, temperature=0):
            '''
            This function is used to assess the significance of the object in the image.
            '''
            def SelfAssessmentTask(file, question, temperature=0):
                answer = self.openai_inference(file, question, temperature)
                with open(file.replace('.jpg','_rank.txt'), 'w') as f:
                    f.write(answer)
            for file in tqdm(files):
                if not os.path.exists(file.replace('.jpg', '_rank.txt')):
                    print(file.replace('.jpg', '_rank.txt'))
                    with open(file.replace('.jpg', '.json'), 'r') as f:
                        reference = json.load(f)['answer']
                    question = ScoreImportancePrompt.format(reference)
                    SelfAssessmentTask(file, question, temperature)
            for file in files:
                with open(file.replace('.jpg','_rank.txt'), 'r') as f:
                    answer = f.read()
                if 'no' in answer.lower() or 'does not' in answer.lower() :
                    score = 0
                else:
                    try:
                        score = int(answer.split("Rating: [[")[1].split("]]")[0])
                    except:
                        try:
                            score = int(answer.split("rating is: [[")[1].split("]]")[0])
                        except:
                            try:
                                score = int(answer.split("[[")[1].split("]]")[0])
                            except:
                                print(f"Missing extract score from {file}")
                                continue
                with open(file.replace('.jpg', '_score.txt'), 'w') as f:
                    f.write(str(score))

        def GeneralInferenceTask(gp):
            '''
            It is used to integrate the results of SelfAssessment and GeneralPerception.
            '''
            text = ''
            files = glob.glob(gp['image'].replace('.jpg', '*').replace('test/images', 'TestObjectAnalyzeImages'))
            for file in files:
                if not '.jpg' in file:
                    continue
                with open(file.replace('.jpg', '_score.txt'), 'r') as f:
                    score = int(f.readline())
                    if score >= 8:  # To modift the threshold, you can change the number here.
                        print(file, 'hello')
                        with open (file.replace('.jpg', '.txt'), 'r') as f:
                            text += f.readline() + ':\n'
                        with open (file.replace('.jpg', '.json'), 'r') as f:
                            data = json.load(f)
                            text += data['answer'] + '\n'
            state = self.openai_inference(
                image=gp['image'],
                question= GeneralPerceptionPrompt.format(text) + gp['question'],
                temperature=0.2,
                max_tokens=500
            )
            gp['answer'] = state
            if not os.path.exists('Outputs'):
                os.makedirs('Outputs')
            with open(os.path.join('Outputs', 'general_perception.jsonl'), 'a') as f:
                f.write(json.dumps(gp) + '\n')    
        
        
        AnalyzeEveryObjectInCODA()
        print('Analyze Every Object In CODA Done')

        SelfAssessmentImages = 'TestObjectAnalyzeImages'

        SelfAssessmentfiles = glob.glob(os.path.join(SelfAssessmentImages, '*.jpg'))
        SelfAssessment(SelfAssessmentfiles)

        print('Self Assessment Done')
        
        with open('CODA-LM/Test/vqa_anno/general_perception.jsonl', 'r') as f0:
            general_perception = [json.loads(line) for line in f0]
            OriginalFiles = [line['image'] for line in general_perception]
            if os.path.exists('Outputs/general_perception.jsonl'):
                OriginalFiles = self.IgnoreGenerated(OriginalFiles, os.path.join('Outputs', 'general_perception.jsonl'))           
            for gp in tqdm(general_perception):
                if gp['image'] not in OriginalFiles:
                    continue
                GeneralInferenceTask(gp)
        self.SortJsonlFile(os.path.join('Outputs', 'general_perception.jsonl'), lambda x: x['question_id'])

    def region_inference(self):
        def regionCategory():
            '''
            Get the category of the object from the CODA dataset. This method will modify the original files by adding category information.
            '''
            categories = {}
            general_situation = {}
            with open('annotations.json', 'r') as f:
                data = json.load(f)
                name = os.getcwd().split('/')[-1]
                for category in data['categories']:
                    categories[category['id']] = category['name']
                for image in data['images']:
                    general_situation[name+'_'+image['file_name'].replace('jpg', 'json')] = {'period': image['period'], 'weather': image['weather']}
                    if 'location' in image:
                        general_situation[name+'_'+image['file_name'].replace('jpg', 'json')]['location'] = image['location']
                inx = 1 
                pre_id = 1
                annotations = []
                name = name.capitalize()
                files = glob.glob('CODA-LM/Test/*.json')
                for file in tqdm(files):
                    with open(file, 'r') as f:
                        data = json.load(f)
                        name = file.split('/')[-1].split('.')[0].split('_')[-1]
                        id = 1000*int(name[0]) + 100*int(name[1]) + 10*int(name[2]) + int(name[3])
                        for d in data['region_perception']:
                            bbox = data['region_perception'][d]['box']
                            result = None
                            with open('annotations.json', 'r') as f:
                                annodata = json.load(f)
                                for anno in annodata['annotations']:
                                    if anno['image_id'] == id and bbox[0] == anno['bbox'][0] and bbox[1] == anno['bbox'][1] and bbox[2] == anno['bbox'][2] and bbox[3] == anno['bbox'][3]:
                                        result = categories[anno['category_id']]
                                        break
                            if result is None:
                                raise NameError
                            data['region_perception'][d]['category_name'] = result
                    with open(file, 'w') as f:
                        json.dump(data, f)
                print('Region Category Done')
        
        regionCategory()
        with open('CODA-LM/Test/vqa_anno/region_perception.jsonl', 'r') as f:
            region_perception = [json.loads(line) for line in f]
            OriginalFiles = [line['image'] for line in region_perception]
            if os.path.exists('Outputs/region_perception.jsonl'):
                OriginalFiles = self.IgnoreGenerated(OriginalFiles, os.path.join('Outputs', 'region_perception.jsonl'))
            for rq in tqdm(region_perception):
                if rq['image'] not in OriginalFiles:
                    continue
                name = os.path.split(rq['image'])[-1].split('_')[0]
                inx = os.path.split(rq['image'])[-1].split('_')[-1].split('.')[0]
                with open('CODA-LM/Test/test_{}.json'.format(name), 'r') as f:
                    data = json.load(f)
                    cate = data['region_perception'][inx]['category_name']
                state = self.openai_inference(
                    image = rq['image'],
                    question = model_prompt + rq['question'] + ' The object in the rectangle is {}.'.format(cate),
                    temperature = 0.2
                )
                rq['answer'] = state + '\n'
                with open(os.path.join('Outputs', 'region_perception.jsonl'), 'a') as f:
                    f.write(json.dumps(rq) + '\n')

        self.SortJsonlFile(os.path.join('Outputs', 'region_perception.jsonl'), lambda x: x['question_id'])

    def driving_suggestions(self):
        def image_title_reg(image_ori_name:str):
            return image_ori_name.split('/')[-1].split('_')[0] + '.jpg'
        def image_title_gen(image_ori_name:str):
            return image_ori_name.split('/')[-1]
        def drivingSuggestionsTask(idx, dsq, gen_prompt, reg_prompt):
            state = self.openai_inference(
                image = dsq['image'],
                question = suggestion_prompt.format(gen_prompt, reg_prompt) + dsq['question'],
                temperature = 0.2
            )
            print(idx)
            dsq['answer'] = state + '\n'
            with open(os.path.join('Outputs', 'driving_suggestion.jsonl'), 'a') as f:
                f.write(json.dumps(dsq) + '\n')
        r_i= 0
        general_perception, region_perception= [], []
        with open(os.path.join('Outputs','general_perception.jsonl'), 'r') as f:
            general_perception = [json.loads(line) for line in f]
        with open(os.path.join('Outputs','region_perception.jsonl'), 'r') as f:
            region_perception = [json.loads(line) for line in f]
        with open('CODA-LM/Test/vqa_anno/driving_suggestion.jsonl', 'r') as f:
            driving_suggestions = [json.loads(line) for line in f]
        OriginalFiles = [line['image'] for line in driving_suggestions]
        if os.path.exists('Outputs/driving_suggestion.jsonl'):
            OriginalFiles = self.IgnoreGenerated(OriginalFiles, os.path.join('Outputs', 'driving_suggestion.jsonl'))
        for idx, dsq in enumerate(tqdm(driving_suggestions)):
            name = image_title_gen(dsq['image'])
            gen_prompt = general_perception[idx]['answer']
            reg_prompt = ''
            while r_i<len(region_perception) and name == image_title_reg(region_perception[r_i]['image']):
                reg_prompt += region_perception[r_i]['answer']
                r_i += 1
            if dsq['image'] not in OriginalFiles:
                continue
            drivingSuggestionsTask(idx, dsq, gen_prompt, reg_prompt)
        self.SortJsonlFile(os.path.join('Outputs', 'driving_suggestion.jsonl'), lambda x: x['question_id'])
    
    def ValidateJsonlFile(self):
        files = glob.glob('Outputs/*.jsonl')
        for file in files:
            with open(file, 'r') as f:
                data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)
            if 'region' in file:
                if df['question_id'].value_counts(sort=True).max() > 1 or df['question_id'].value_counts(sort=True).count() != 1123:
                    print('Validate Failed!!!')
                    print('Duplicate in {}'.format(file))
            else:
                if df['question_id'].value_counts(sort=True).max() > 1 or df['question_id'].value_counts(sort=True).count() != 500:
                    print('Validate Failed!!!')
                    print('Duplicate in {}'.format(file))
        print('Validation Done!!!')


    def run(self):
        self.region_inference()
        print('Region Inference Done')
        self.general_inference()
        print('General Inference Done')
        self.driving_suggestions()
        print('Driving Suggestions Done')
        self.ValidateJsonlFile()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-key", required=True)
    parser.add_argument("--model",  required=True, type=str)
    parser.add_argument("--debug-num", default=100, type=int, help="Only use for mode 1 & 2")
    parser.add_argument('--base-url', type=str)
    args = parser.parse_args()
    Application = CODALMTask(args)
    Application.run()
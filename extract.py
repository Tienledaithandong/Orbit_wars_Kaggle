import json

def extract():
    with open('c:/LENAMTIEN/KAGGLE/ORBIT_WARS/other_agent_ideas/orbit-wars-reinforcement-learning-tutorial.ipynb', encoding='utf-8') as f:
        d = json.load(f)
    code = ["".join(cell['source']) for cell in d['cells'] if cell['cell_type'] == 'code']
    with open('c:/LENAMTIEN/KAGGLE/ORBIT_WARS/scratch_extracted.py', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(code))

if __name__ == '__main__':
    extract()
